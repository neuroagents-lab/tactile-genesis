"""Task-space action terms: differential IK and operational-space control."""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import torch

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions


@ACTION_TERM_REGISTRY.register()
class DifferentialIKController(ActionTerm):
    """Action term that maps task-space commands to joint targets via differential IK.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: list[str]
        The names of the DOFs to control.
    ee_link_name: str
        The name of the end-effector link.
    scale: float | list[float]
        The scale to apply to the actions (x, y, z, rx, ry, rz).
    """

    ee_link_name: str = ""
    scale: float | list[float] = 0.5

    def __init__(
        self,
        env: EnvBase,
        options: ActionTermOptions,
    ):
        super().__init__(env=env, options=options)
        self._ee_link = None
        self.controller_damping = 1e-4
        self._diag = None
        self._scale = None

    def build(self) -> None:
        super().build()

        self._ee_link = self.entity.get_link(self.ee_link_name)

        # Jacobian columns are now in dofs_name order; build indices to select
        # this action's DOF subset from the reordered Jacobian.
        base_offset = 0 if self.entity.is_fixed_base else 6
        self._jac_dofs_idx = torch.as_tensor(
            [base_offset + self.entity.dofs_idx_map[idx] for idx in self.dofs_idx_local.tolist()],
            dtype=gs.tc_int,
            device=self.device,
        )

        if isinstance(self.scale, float):
            self._scale = float(self.scale)
        elif isinstance(self.scale, dict):
            self._scale = torch.tensor(self.scale, device=self.device).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported scale type: {type(self.scale)}.")

    @property
    def action_dim(self):
        return 6  # (x, y, z, rx, ry, rz)

    @property
    def diag(self):
        if self._diag is None:
            self._diag = self.controller_damping * torch.eye(6, device=self.device, dtype=gs.tc_float)
        return self._diag

    def compute(self, actions: torch.Tensor) -> None:
        assert actions.shape == (
            self.num_envs,
            self.action_dim,
        ), f"Expected actions shape: (num_envs, {self.action_dim}), got {actions.shape=}"
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions

        delta_target = self._scale * torch.tanh(actions)
        J = self.entity.get_jacobian(link=self._ee_link)[:, :, self._jac_dofs_idx]  # (n_envs, 6, n_dofs)
        J_T = J.transpose(1, 2)  # (n_envs, n_dofs, 6)
        delta_q = J_T @ torch.linalg.solve(J @ J_T + self.diag, delta_target).unsqueeze(-1)
        delta_q = delta_q.squeeze(-1)  # (n_envs, n_dofs)

        self._processed_action = delta_q + self.entity.get_dofs_pos(dofs_idx_local=self.dofs_idx_local)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0

    def apply_actions(self) -> None:
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=self.dofs_idx_local)


@ACTION_TERM_REGISTRY.register()
class OperationalSpaceController(ActionTerm):
    """
    Operational Space Control (OSC) for either position only or position and orientation.

    Reference:
    ----------
    [1] http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
    """

    ee_link_name: str = ""
    scale: float | list[float] = 0.5
    kp: float = 100
    kd: float = 10
    position_only: bool = False

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        super().__init__(env=env, options=options)
        self.ee_link_idx: torch.Tensor | None = None
        self._ee_link = None
        self.default_dofs_pos: torch.Tensor | None = None

        self.uncoupling = True
        self.diag: torch.Tensor | None = None
        self._scale: float | torch.Tensor | None = None

    def build(self) -> None:
        super().build()

        _, ee_link_idx = self.entity.find_named_links_idx_local(self.ee_link_name)
        self.ee_link_idx = torch.as_tensor(ee_link_idx, dtype=gs.tc_int, device=self.device).contiguous()
        self._ee_link = self.entity.get_link(self.ee_link_name)
        dofs_idx = [self.entity.dofs_idx_map[idx] for idx in self.dofs_idx_local.tolist()]
        self.default_dofs_pos = self.entity.default_dofs_pos[:, dofs_idx].clone()

        # Jacobian columns are now in dofs_name order; build indices to select
        # this action's DOF subset from the reordered Jacobian.
        base_offset = 0 if self.entity.is_fixed_base else 6
        self._jac_dofs_idx = torch.as_tensor(
            [base_offset + idx for idx in dofs_idx],
            dtype=gs.tc_int,
            device=self.device,
        )

        if self.position_only:
            if isinstance(self.scale, float):
                self._scale = self.scale
            elif isinstance(self.scale, list):
                assert len(self.scale) == 3, f"Expected scale length 3, got {len(self.scale)}"
                self._scale = torch.tensor(self.scale, device=self.device).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported scale type: {type(self.scale)}.")
        else:
            if isinstance(self.scale, float):
                self._scale = self.scale
            elif isinstance(self.scale, list):
                assert len(self.scale) == 6, f"Expected scale length 6, got {len(self.scale)}"
                self._scale = torch.tensor(self.scale, device=self.device).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported scale type: {type(self.scale)}.")

    @property
    def action_dim(self):
        if self.position_only:
            return 3
        else:
            return 6

    def compute(self, actions: torch.Tensor) -> None:
        """
        Compute the torques based on the desired joint positions.

        Parameters
        ----------
        actions: torch.Tensor
            shape (num_envs, 3+3) pos:3 quat:3
        """
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions

        self.ee_lin_vel = self.entity.get_links_vel(ls_idx_local=self.ee_link_idx).squeeze(1)
        self.ee_ang_vel = self.entity.get_links_ang(ls_idx_local=self.ee_link_idx).squeeze(1)

        # NOTE: scale the input to make sure delta is small enough
        delta_target = self._scale * torch.tanh(actions)

        if self.entity.get_vel() is not None:
            vel_pos_error = -(self.ee_lin_vel - self.entity.get_vel())
        else:
            vel_pos_error = -self.ee_lin_vel

        # F_r = kp * pos_err + kd * vel_err
        desired_force = self.kp * delta_target[:, :3] + self.kd * vel_pos_error

        if self.entity.get_ang() is not None:
            vel_ori_error = -(self.ee_ang_vel - self.entity.get_ang())
        else:
            vel_ori_error = -self.ee_ang_vel

        # Tau_r = kp * ori_err + kd * vel_err
        if self.position_only:
            desired_torque = self.kd * vel_ori_error
        else:
            desired_torque = self.kp * delta_target[:, 3:] + self.kd * vel_ori_error

        J = self.entity.get_jacobian(link=self._ee_link)[:, :, self._jac_dofs_idx]  # (n_envs, 6, n_dofs)
        mass_mat_L, mass_mat_D_inv = self.entity.get_mass_mat(decompose=True)
        mass_mat_L = mass_mat_L[:, self.dofs_idx_local][:, :, self.dofs_idx_local]
        mass_mat_D_inv = mass_mat_D_inv[:, self.dofs_idx_local]
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = self.compute_ops_kinetics_cholesky(
            mass_mat_L, mass_mat_D_inv, J
        )

        # Decouples desired positional control from orientation control
        if self.uncoupling:
            decoupled_force = torch.matmul(lambda_pos, desired_force.unsqueeze(-1))
            decoupled_torque = torch.matmul(lambda_ori, desired_torque.unsqueeze(-1))
            decoupled_wrench = torch.cat([decoupled_force, decoupled_torque], dim=1)
        else:
            desired_wrench = torch.cat([desired_force, desired_torque], dim=1)
            decoupled_wrench = torch.matmul(lambda_full, desired_wrench.unsqueeze(-1))

        # Gamma (without null torques) = J^T * F + gravity compensations
        # NOTE: use material=gs.materials.Rigid(gravity_compensation=1.0) for now
        # TODO: need to add gravity compensations torque
        ctl_torques = torch.matmul(J.transpose(-2, -1), decoupled_wrench).squeeze(
            -1
        )  # TODO: + self.torque_compensation

        mass_mat = (mass_mat_L.transpose(-2, -1) * mass_mat_D_inv.unsqueeze(1)) @ mass_mat_L
        # Calculate and add nullspace torques
        ctl_torques += self.nullspace_torques(
            mass_mat,
            nullspace_matrix,
            self.default_dofs_pos,
            self.entity.get_dofs_pos(dofs_idx_local=self.dofs_idx_local),
            self.entity.get_dofs_vel(dofs_idx_local=self.dofs_idx_local),
        )

        self._processed_action = ctl_torques

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0

    def apply_actions(self) -> None:
        self.entity.control_dofs_force(self._processed_action, dofs_idx_local=self.dofs_idx_local)

    @staticmethod
    def compute_ops_kinetics_cholesky(
        mass_mat_U: torch.Tensor,
        mass_mat_D_inv_vec: torch.Tensor,
        J: torch.Tensor,
        is_U_unitriangular: bool = True,
    ):
        """Compute operational space kinetics from a precomputed U^T D U mass-matrix decomposition.

        The decomposition is ``M = U_factor^T D U_factor``. This method leverages
        the precomputed factors for efficiency and stability.

        Parameters
        ----------
        mass_mat_U: torch.Tensor
            The U factor (typically unit upper triangular) from M = U^T D U.  Shape (B, N, N) or (N, N).
        mass_mat_D_inv_vec: torch.Tensor
            Vector of diagonal elements of D^-1 (the inverse of the diagonal matrix D). Shape (B, N) or (N,).
        J: torch.Tensor
            The full Jacobian (B, C, N) or (C, N). C is the total number of constraints.
        is_U_unitriangular: bool
            True if mass_mat_U is unit triangular. This is common in Pinocchio-like libraries.

        Returns
        -------
        lambda_full: torch.Tensor
            Full operational space inertia (Lambda_full).
        lambda_pos: torch.Tensor
            Positional part of operational space inertia (Lambda_pos).
        lambda_ori: torch.Tensor
            Orientational part of operational space inertia (Lambda_ori).
        N_proj: torch.Tensor
            Nullspace projection matrix.
        """
        is_batched_input = mass_mat_U.ndim == 3
        if not is_batched_input:
            # Unsqueeze to add a batch dimension for consistent processing
            mass_mat_U = mass_mat_U.unsqueeze(0)
            mass_mat_D_inv_vec = mass_mat_D_inv_vec.unsqueeze(0)
            J = J.unsqueeze(0)

        B, _, N_dof = J.shape

        # L_factor_from_U is U_factor^T. If U_factor is unit upper, L_factor_from_U is unit lower.
        L_factor_from_U = mass_mat_U.transpose(-2, -1)

        # Full Jacobian
        J_T_full = J.transpose(-2, -1)  # (B, N_dof, C_total)
        # K_full = L_factor_from_U^-1 @ J_T_full
        K_full = torch.linalg.solve_triangular(
            L_factor_from_U, J_T_full, upper=False, unitriangular=is_U_unitriangular
        )  # (B, N_dof, C_total)

        # lambda_full_inv = K_full^T @ diag(D_inv_vec) @ K_full
        # Efficient computation: (K_full^T * D_inv_vec_row_for_K_T) @ K_full
        lambda_full_inv = (K_full.transpose(-2, -1) * mass_mat_D_inv_vec.unsqueeze(1)) @ K_full  # (B, C_total, C_total)

        # Positional part
        J_pos_T = J[:, :3, :].transpose(-2, -1)  # (B, N_dof, 3)
        K_pos = torch.linalg.solve_triangular(
            L_factor_from_U, J_pos_T, upper=False, unitriangular=is_U_unitriangular
        )  # (B, N_dof, 3)
        lambda_pos_inv = (K_pos.transpose(-2, -1) * mass_mat_D_inv_vec.unsqueeze(1)) @ K_pos  # (B, 3, 3)

        # Orientational part
        J_ori = J[..., 3:, :]  # (B, C_ori, N_dof) where C_ori = C_total - 3
        J_ori_T = J_ori.transpose(-2, -1)  # (B, N_dof, C_ori)
        K_ori = torch.linalg.solve_triangular(
            L_factor_from_U, J_ori_T, upper=False, unitriangular=is_U_unitriangular
        )  # (B, N_dof, C_ori)
        lambda_ori_inv = (K_ori.transpose(-2, -1) * mass_mat_D_inv_vec.unsqueeze(1)) @ K_ori  # (B, C_ori, C_ori)

        # Take pseudo-inverses for numerical stability: Lambda = (J M^-1 J^T)^+
        lambda_full = torch.linalg.pinv(lambda_full_inv)
        lambda_pos = torch.linalg.pinv(lambda_pos_inv)
        # Handle case where J_ori might be empty (e.g., if C_total <= 3)
        if J_ori.shape[-2] > 0:  # If C_ori > 0
            lambda_ori = torch.linalg.pinv(lambda_ori_inv)
        else:
            # Create an empty tensor with the correct batch dimension and device/dtype
            # Expected shape for lambda_ori is (B, 0, 0) or (0,0) if not batched
            lambda_ori = torch.empty((B, 0, 0), device=J.device, dtype=J.dtype)

        # Y = D_inv K_full
        Y = mass_mat_D_inv_vec.unsqueeze(-1) * K_full  # (B, N_dof, C_total)

        # Z_jbar is M^-1 @ J^T
        Z_jbar = torch.linalg.solve_triangular(
            mass_mat_U, Y, upper=True, unitriangular=is_U_unitriangular
        )  # (B, N_dof, C_total)

        # Jbar = (M^-1 @ J^T) @ Lambda_full
        Jbar = torch.matmul(Z_jbar, lambda_full)  # (B, N_dof, C_total)

        # Nullspace projection matrix: N_proj = I - J_bar @ J
        I_N_dof = torch.eye(N_dof, device=J.device, dtype=J.dtype).unsqueeze(0).expand(B, -1, -1)
        N_proj = I_N_dof - torch.matmul(Jbar, J)  # (B, N_dof, N_dof)

        if not is_batched_input:
            lambda_full = lambda_full.squeeze(0)
            lambda_pos = lambda_pos.squeeze(0)
            lambda_ori = lambda_ori.squeeze(0) if J_ori.shape[-2] > 0 else lambda_ori  # Squeeze if not already (0,0)
            N_proj = N_proj.squeeze(0)

        return lambda_full, lambda_pos, lambda_ori, N_proj

    @staticmethod
    def nullspace_torques(
        mass_matrix: torch.Tensor,  # [B, N, N]
        nullspace_matrix: torch.Tensor,  # [B, N, N] (assuming M=N for (I-Jbar J))
        initial_joint: torch.Tensor,  # [B, N]
        joint_pos: torch.Tensor,  # [B, N]
        joint_vel: torch.Tensor,  # [B, N]
        joint_kp: float = 10.0,
    ) -> torch.Tensor:
        """Compute nullspace torques for a robot with redundant DOFs.

        For a robot with redundant DOF(s), a nullspace exists which is orthogonal to the remainder of the controllable
        subspace of the robot's joints. Therefore, an additional secondary objective that does not impact the original
        controller objective may attempt to be maintained using these nullspace torques.

        Parameters
        ----------
        mass_matrix : torch.Tensor
            2d array representing the mass matrix of the robot
        nullspace_matrix : torch.Tensor
            2d array representing the nullspace matrix of the robot
        initial_joint : torch.Tensor
            Joint configuration to be used for calculating nullspace torques
        joint_pos : torch.Tensor
            Current joint positions
        joint_vel : torch.Tensor
            Current joint velocities
        joint_kp : float
            Proportional control gain when calculating nullspace torques

        Returns
        -------
        torques: torch.Tensor
            Nullspace torques
        """
        joint_kv = 2.0 * joint_kp**0.5

        # Compute pose torques: [B, N]
        # q_ddot_null = kp * (q_initial - q_current) - kv * q_dot_current
        pose_error = joint_kp * (initial_joint - joint_pos) - joint_kv * joint_vel
        # Gamma_null = M @ q_ddot_null
        # mass_matrix: [B,N,N], pose_error.unsqueeze(-1): [B,N,1] -> desired_nullspace_joint_torques: [B,N,1]
        desired_nullspace_joint_torques = torch.matmul(mass_matrix, pose_error.unsqueeze(-1))

        # Project into nullspace: torques_nullspace = (I - Jbar*J)^T @ Gamma_null
        # nullspace_matrix.transpose(1, 2) is [B, N, N] (transpose of each matrix in batch)
        return torch.matmul(nullspace_matrix.transpose(1, 2), desired_nullspace_joint_torques).squeeze(-1)
