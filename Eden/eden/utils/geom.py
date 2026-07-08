"""Geometry helpers: axis alignment, point transforms, rotation conversions."""

import genesis as gs
from genesis.utils.geom import (
    rot6d_to_R,
    R_to_quat,
    quat_to_R,
    R_to_rot6d,
    transform_by_quat,
    normalize,
)
import numpy as np
import torch


def align_up_to_z(up, as_torch=False):
    """
    Aligns the given 'up' vector to the Z-up direction (0, 0, 1).

    Parameters
    ----------
    up : list | tuple | torch.Tensor
        A 3D vector to align to Z-up. If list/tuple, it's converted to a tensor.
    as_torch : bool
        If True, the returned rotation matrix is a torch.Tensor. Otherwise, it's a numpy.ndarray.

    Returns
    -------
    R : torch.Tensor | np.ndarray
        A 3x3 rotation matrix that aligns 'up' to the Z-axis.
    """
    if as_torch:
        if not isinstance(up, torch.Tensor):
            up = torch.tensor(up, dtype=gs.tc_float, device=gs.device)

        up = up / torch.linalg.norm(up)

        target = torch.tensor([0.0, 0.0, 1.0], dtype=gs.tc_float, device=gs.device)
        if torch.allclose(up, target, atol=1e-6):
            return torch.eye(3, dtype=gs.tc_float, device=gs.device)

        if torch.allclose(up, -target, atol=1e-6):
            return torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                dtype=gs.tc_float,
                device=gs.device,
            )

        axis = torch.cross(up, target)
        axis = axis / torch.linalg.norm(axis)

        dot_product = torch.dot(up, target)
        angle = torch.acos(torch.clamp(dot_product, min=-1.0, max=1.0))

        K = torch.zeros((3, 3), dtype=gs.tc_float, device=gs.device)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]

        R = torch.eye(3, dtype=gs.tc_float, device=gs.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return R

    else:
        up = np.array(up, dtype=np.float64)
        up /= np.linalg.norm(up)  # Normalize the input vector

        target = np.array([0, 0, 1])  # Z-up direction
        if np.allclose(up, target):  # Already aligned
            return np.eye(3)

        if np.allclose(up, -target):  # Opposite direction case
            # Rotate 180 degrees around an arbitrary perpendicular axis
            return np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])  # 180-degree rotation around X or Y works

        axis = np.cross(up, target)
        angle = np.arccos(np.dot(up, target))

        # Normalize axis
        axis /= np.linalg.norm(axis)

        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R


def align_up_and_front(up, front, as_torch=False):
    """Return a rotation matrix R that realigns an asset's canonical axes to world axes.

    The asset's canonical up and front axes are given by `up` and `front` (in the
    asset's own mesh frame) so that after applying R::

        R @ up    = (0, 0, 1)   (asset's up aligned to world Z)
        R @ front = (1, 0, 0)   (asset's front aligned to world X)

    Parameters
    ----------
    up : list | tuple | torch.Tensor
        The asset's "up" direction in its own canonical/mesh frame.
    front : list | tuple | torch.Tensor
        The asset's "front" direction in its own canonical/mesh frame.
    as_torch : bool
        If True, the returned rotation matrix is a torch.Tensor. Otherwise, it's a numpy.ndarray.

    Returns
    -------
    R : torch.Tensor | np.ndarray
        A 3x3 rotation matrix.
    """
    if as_torch:
        # Convert inputs to PyTorch tensors with specified dtype and device
        if not isinstance(up, torch.Tensor):
            up = torch.tensor(up, dtype=gs.tc_float, device=gs.device)
        if not isinstance(front, torch.Tensor):
            front = torch.tensor(front, dtype=gs.tc_float, device=gs.device)

        norm_up = torch.linalg.norm(up)
        up = up / norm_up

        norm_front = torch.linalg.norm(front)
        front = front / norm_front

        R1 = align_up_to_z(up, as_torch=True)

        front_transformed = R1 @ front

        front_xy = front_transformed.clone()
        front_xy[2] = 0.0

        norm_front_xy = torch.linalg.norm(front_xy)
        if norm_front_xy < 1e-6:  # Tolerance for norm being zero
            return R1

        front_xy /= norm_front_xy  # Normalize the projected vector

        angle_of_front_xy = torch.atan2(front_xy[1], front_xy[0])

        cos_val = torch.cos(-angle_of_front_xy)
        sin_val = torch.sin(-angle_of_front_xy)

        R2 = torch.eye(3, dtype=gs.tc_float, device=gs.device)
        R2[0, 0] = cos_val
        R2[0, 1] = -sin_val
        R2[1, 0] = sin_val
        R2[1, 1] = cos_val

        return R2 @ R1

    else:
        up = np.array(up, dtype=np.float64)
        front = np.array(front, dtype=np.float64)

        # Normalize the vectors
        up /= np.linalg.norm(up)
        front /= np.linalg.norm(front)

        # First, align "up" to (0,0,1)
        R1 = align_up_to_z(up, as_torch=False)

        # Rotate the front vector using the first transformation
        front_transformed = R1 @ front

        # Project the transformed front vector onto the XY plane
        front_xy = front_transformed.copy()
        front_xy[2] = 0

        norm_front_xy = np.linalg.norm(front_xy)
        if norm_front_xy < 1e-6:  # If front is vertical, default to (1,0,0)
            return R1

        front_xy /= norm_front_xy  # Normalize projection

        # Compute angle to rotate front_xy to (1,0,0)
        angle = np.arctan2(front_xy[1], front_xy[0])  # Angle in XY plane

        # Construct a Z-axis rotation matrix
        R2 = np.array(
            [
                [np.cos(-angle), -np.sin(-angle), 0],
                [np.sin(-angle), np.cos(-angle), 0],
                [0, 0, 1],
            ]
        )

        # Final transformation: First align up to Z, then rotate front to X
        return R2 @ R1


@torch.jit.script
def transform_by_R(pos: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Transform 3D points by a 3x3 rotation matrix or a batch of matrices.

    Supports both NumPy arrays and PyTorch tensors.

    Parameters
    ----------
    pos: torch.Tensor
        A torch tensor of 3D points. Can be a single point
         (3,), a batch of points (B, 3), or a batched batch of points (B, N, 3).
    R: torch.Tensor
        The 3x3 rotation matrix or a batch of B rotation
        matrices of shape (B, 3, 3). Must be of the same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    assert pos.shape[-1] == 3

    dim_added = False
    if R.ndim == 2:
        R = R[None]
        dim_added = True
    if pos.ndim == 3:
        new_pos = (R @ pos.swapaxes(-1, -2)).swapaxes(-1, -2)
    elif pos.ndim == 2:
        new_pos = (R @ pos[:, :, None])[..., 0]
    else:
        new_pos = (R @ pos[None, :, None])[..., 0]
        if dim_added:
            new_pos = new_pos[0]
    return new_pos


@torch.jit.script
def transform_by_T(pos: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Transform 3D points by a 4x4 transformation matrix or a batch of matrices.

    Supports both NumPy arrays and PyTorch tensors.

    Parameters
    ----------
    pos: np.ndarray | torch.Tensor
        A numpy array or torch tensor of 3D points. Can be a single point
         (3,), a batch of points (B, 3), or a batched batch of points (B, N, 3).
    T: np.ndarray | torch.Tensor
        The 4x4 transformation matrix or a batch of B transformation
        matrices of shape (B, 4, 4). Must be of the same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    assert pos.shape[-1] == 3, "Input positions must have 3 dimensions"

    if T.ndim == 2:
        T = T.reshape(1, 4, 4)

    if pos.ndim > 1:
        ones_shape = pos.shape[:-1] + (1,)
        pos_hom = torch.cat([pos, torch.ones(ones_shape, dtype=pos.dtype, device=pos.device)], dim=-1)
    else:
        pos_hom = torch.cat([pos, torch.tensor([1.0], dtype=pos.dtype, device=pos.device)])

    if pos_hom.ndim == 1:
        pos_hom = pos_hom.reshape(1, 1, -1)
    elif pos_hom.ndim == 2:
        assert T.shape[0] == 1 or T.shape[0] == pos.shape[0], f"{T.shape}, {pos.shape}"
        pos_hom = pos_hom.reshape(-1, 1, 4)

    pos_hom_t = pos_hom.swapaxes(-1, -2)  # (..., N, 4) -> (..., 4, N)
    transformed_hom = T @ pos_hom_t
    transformed_hom = transformed_hom.swapaxes(-1, -2)[..., :3]

    if pos.ndim == 1:
        transformed_hom = transformed_hom.reshape(-1)
    elif pos.ndim == 2:
        transformed_hom = transformed_hom.reshape(-1, 3)
    return transformed_hom


def rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
    R = rot6d_to_R(d6)
    return R_to_quat(R)


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    R = quat_to_R(quat)
    return R_to_rot6d(R)


def transform_by_quat_yaw(v: torch.Tensor, quat: torch.Tensor):
    quat_yaw = quat.clone().detach().view(-1, 4)
    quat_yaw[..., 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return transform_by_quat(v, quat_yaw)


def xyzw_to_wxyz(q):
    """Reorder the last axis of a quaternion from (x, y, z, w) to (w, x, y, z).

    Accepts either ``numpy.ndarray`` or ``torch.Tensor``; fancy indexing on the
    last axis is backend-agnostic.
    """
    return q[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q):
    """Reorder the last axis of a quaternion from (w, x, y, z) to (x, y, z, w).

    Accepts either ``numpy.ndarray`` or ``torch.Tensor``.
    """
    return q[..., [1, 2, 3, 0]]
