"""Viser-based interactive visualizer for Eden environments."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
import viser
from genesis.options.morphs import MJCF, URDF, USD, Mesh, Plane, Primitive
from genesis.utils.geom import trans_quat_to_T

import eden as en

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink


_DEFAULT_FOV_DEGREES = 60
_DEFAULT_FOV_MIN = 20
_DEFAULT_FOV_MAX = 150


class LinkHandle:
    def __init__(
        self,
        handle: viser.SceneNodeHandle,
        link: RigidLink,
        canonical_pos: torch.Tensor,
        canonical_quat: torch.Tensor,
        active_envs_idx: torch.Tensor | list | None = None,
    ):
        self.handle = handle
        self.link = link
        self.canonical_pos = canonical_pos
        self.canonical_quat = canonical_quat
        self.active_envs_idx = active_envs_idx

    def update(self, env_offset: np.ndarray | None = None):
        pos = self.link.get_pos()
        quat = self.link.get_quat()

        if self.active_envs_idx is not None:
            pos = pos[self.active_envs_idx]
            quat = quat[self.active_envs_idx]

        pos = pos.cpu().numpy()
        if env_offset is not None:
            offset = env_offset[self.active_envs_idx] if self.active_envs_idx is not None else env_offset
            self.handle.batched_positions = pos + offset
        else:
            self.handle.batched_positions = pos
        self.handle.batched_wxyzs = quat.cpu().numpy()


class ViserViewer:
    """Viser visualizer for Eden.

    This extension allows visualizing Eden environments using the Viser library.
    """

    def __init__(
        self,
        env: EnvBase,
        host: str = "localhost",
        port: int = 8080,
        enable_gui: bool = True,
    ):
        self.env = env
        self.host = host
        self.port = port
        self.enable_gui = enable_gui

        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        self.gui_elements = {}

        self._entity_handles: dict[str, LinkHandle] = {}
        self._collision_entity_handles: dict[str, LinkHandle] = {}
        self._handles_lock = threading.Lock()

        self.collision_mesh_enabled = False

        self._built = False

        self._gui_event_state = {}

    def build(self):
        """Build the visualization scene from the Eden environment."""
        if self._built:
            return

        if self.enable_gui:
            self._setup_gui()

        for name, entity in self.env.entities.items():
            self.add_entity(name, entity)

        self._built = True

    @property
    def envs_offset(self) -> np.ndarray:
        return self.env.scene.envs_offset

    def _setup_gui(self):
        """Set up default GUI elements."""
        with self.server.gui.add_folder("Simulation"):
            self.gui_elements["reset"] = self.server.gui.add_button("Reset Env")
            # self.gui_elements["reset"].on_click(lambda _: self.env.reset())

            @self.gui_elements["reset"].on_click
            def _(_):
                self._gui_event_state["reset_triggered"] = True
                self.env.reset()

        with self.server.gui.add_folder("Visualization"):
            slider_fov = self.server.gui.add_slider(
                "FOV (°)",
                min=_DEFAULT_FOV_MIN,
                max=_DEFAULT_FOV_MAX,
                step=1,
                initial_value=_DEFAULT_FOV_DEGREES,
                hint="Vertical FOV of viewer camera, in degrees.",
            )

            @slider_fov.on_update
            def _(_) -> None:
                for client in self.server.get_clients().values():
                    client.camera.fov = np.radians(slider_fov.value)

            @self.server.on_client_connect
            def _(client: viser.ClientHandle) -> None:
                client.camera.fov = np.radians(slider_fov.value)

            cb_collision_mesh = self.server.gui.add_checkbox(
                "Collision Mesh",
                initial_value=self.collision_mesh_enabled,
                hint="Show collision meshes.",
            )

            @cb_collision_mesh.on_update
            def _(_) -> None:
                self.collision_mesh_enabled = cb_collision_mesh.value

                # Copy to avoid iterating while updating
                with self._handles_lock:
                    entity_handles = list(self._entity_handles.values())
                    collision_handles = list(self._collision_entity_handles.values())
                    needs_collision_meshes = not self._collision_entity_handles

                for link_handle in entity_handles:
                    link_handle.handle.visible = not self.collision_mesh_enabled

                for link_handle in collision_handles:
                    link_handle.handle.visible = self.collision_mesh_enabled

                if needs_collision_meshes:
                    for name, entity in self.env.entities.items():
                        self.add_entity(name, entity, use_visual_mesh=False)

    def _create_batched_mesh_handle(
        self,
        name: str,
        mesh: trimesh.Trimesh,
        num_instances: int,
    ) -> viser.SceneNodeHandle:
        """Create a batched mesh handle with default poses."""
        return self.server.scene.add_batched_meshes_trimesh(
            name=name,
            mesh=mesh,
            batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(num_instances, axis=0),
            batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(num_instances, axis=0),
        )

    def _store_handle(
        self,
        name: str,
        link_handle: LinkHandle,
        use_visual_mesh: bool,
    ):
        """Store a link handle in the appropriate dictionary."""
        with self._handles_lock:
            if use_visual_mesh:
                self._entity_handles[name] = link_handle
            else:
                self._collision_entity_handles[name] = link_handle

    def add_entity(self, name: str, entity: Entity, use_visual_mesh: bool = True):
        # Check if entity is heterogeneous (grouped entity with multiple morphs)
        if entity.is_heterogeneous:
            self._add_heterogeneous_entity(name, entity, use_visual_mesh)
        elif isinstance(entity.morph, (URDF, MJCF, USD)):
            self._add_articulated_entity(name, entity, use_visual_mesh)
        elif isinstance(entity.morph, Plane):
            self._add_mesh_entity(name, entity, use_visual_mesh, is_batched=False)
        elif isinstance(entity.morph, (Mesh, Primitive)):
            self._add_mesh_entity(name, entity, use_visual_mesh)
        else:
            raise ValueError(f"Entity type not supported: {type(entity.morph)}")

    def _get_geoms(self, link, use_visual_mesh: bool, allow_collision_as_visual: bool):
        if use_visual_mesh:
            if link.vgeoms:
                return link.vgeoms
            if allow_collision_as_visual and link.geoms:
                en.logger.warning(
                    f"[Viser visualization] Link '{link.name}' has no visual geometry. "
                    "Using collision geometry as fallback."
                )
                return link.geoms
            return []
        return link.geoms

    def _add_heterogeneous_entity(
        self, name: str, entity: Entity, use_visual_mesh: bool = True, allow_collision_as_visual: bool = False
    ):
        """Add a heterogeneous entity (grouped entity with multiple morphs) to the Viser scene."""
        assert entity.is_heterogeneous and entity._entity._enable_heterogeneous

        link = entity.links[0]
        geoms = self._get_geoms(link, use_visual_mesh, allow_collision_as_visual)
        prefix = "visual" if use_visual_mesh else "collision"
        if len(geoms) == 0:
            kind = "visual" if use_visual_mesh else "collision"
            en.logger.warning(f"[Viser visualization] Link '{link.name}' has no {kind} geometry. Skipping.")
            return

        # Genesis heterogeneous entities have all variants' geoms in a single link
        for i, geom in enumerate(geoms):
            handle_name = f"/{prefix}/{name}_geom_{i}"

            active_envs_idx = (
                geom.active_envs_idx if hasattr(geom, "active_envs_idx") and geom.active_envs_idx is not None else None
            )
            num_instances = len(active_envs_idx) if active_envs_idx is not None else self.env.num_envs

            mesh = geom.get_trimesh().copy()
            geom_pos = torch.from_numpy(geom.init_pos).float()
            geom_quat = torch.from_numpy(geom.init_quat).float()
            T = trans_quat_to_T(geom_pos.unsqueeze(0), geom_quat.unsqueeze(0))
            mesh.apply_transform(T[0])

            viser_handle = self._create_batched_mesh_handle(handle_name, mesh, num_instances)

            link_handle = LinkHandle(
                handle=viser_handle,
                link=link,
                canonical_pos=geom.get_pos().clone().detach(),
                canonical_quat=geom.get_quat().clone().detach(),
                active_envs_idx=active_envs_idx,
            )
            self._store_handle(handle_name, link_handle, use_visual_mesh)

    def _build_combined_mesh(self, geoms):
        meshes = []
        for geom in geoms:
            mesh = geom.get_trimesh().copy()
            geom_pos = torch.from_numpy(geom.init_pos).float()
            geom_quat = torch.from_numpy(geom.init_quat).float()
            T = trans_quat_to_T(geom_pos.unsqueeze(0), geom_quat.unsqueeze(0))
            mesh.apply_transform(T[0])
            meshes.append(mesh)
        return trimesh.util.concatenate(meshes)

    def _add_entity_common(
        self,
        name: str,
        entity: Entity,
        use_visual_mesh: bool,
        allow_collision_as_visual: bool,
        num_envs: int,
        active_envs_idx=None,
    ):
        prefix = "visual" if use_visual_mesh else "collision"

        for link in entity.links:
            geoms = self._get_geoms(link, use_visual_mesh, allow_collision_as_visual)

            if len(geoms) == 0:
                kind = "visual" if use_visual_mesh else "collision"
                en.logger.warning(f"[Viser visualization] Link '{link.name}' has no {kind} geometry. Skipping.")
                continue

            combined = self._build_combined_mesh(geoms)
            handle_name = f"/{prefix}/{name}_{link.name}"

            viser_handle = self._create_batched_mesh_handle(handle_name, combined, num_envs)

            link_handle = LinkHandle(
                handle=viser_handle,
                link=link,
                canonical_pos=link.get_pos().clone().detach(),
                canonical_quat=link.get_quat().clone().detach(),
                active_envs_idx=active_envs_idx,
            )

            self._store_handle(handle_name, link_handle, use_visual_mesh)

    def _add_mesh_entity(
        self,
        name: str,
        entity: Entity,
        use_visual_mesh: bool = True,
        is_batched: bool = True,
        allow_collision_as_visual: bool = False,
    ):
        self._add_entity_common(
            name=name,
            entity=entity,
            use_visual_mesh=use_visual_mesh,
            allow_collision_as_visual=allow_collision_as_visual,
            num_envs=self.env.num_envs if is_batched else 1,
            active_envs_idx=None if is_batched else [0],
        )

    def _add_articulated_entity(
        self,
        name: str,
        entity: Entity,
        use_visual_mesh: bool = True,
        allow_collision_as_visual: bool = False,
    ):
        qs_idx_local = slice(None)

        tmp_qpos = entity.get_qpos(qs_idx_local=qs_idx_local).clone().detach()

        canonical_qpos = torch.zeros_like(tmp_qpos)
        if not entity.is_fixed_base:
            if canonical_qpos.ndim == 1:  # single instance
                canonical_qpos[3] = 1.0
            else:
                canonical_qpos[:, 3] = 1.0  # w component of quaternion

        entity.set_qpos(canonical_qpos, qs_idx_local=qs_idx_local)

        self._add_entity_common(
            name=name,
            entity=entity,
            use_visual_mesh=use_visual_mesh,
            allow_collision_as_visual=allow_collision_as_visual,
            num_envs=self.env.num_envs,
            active_envs_idx=None,
        )

        entity.set_qpos(tmp_qpos, qs_idx_local=qs_idx_local)

    def add_point_cloud(self, name: str, points: np.ndarray, colors: np.ndarray):
        self.server.scene.add_point_cloud(
            name=name,
            points=points,
            colors=colors,
            point_size=0.001,
        )

    def update(self):
        """Update the visualization with current simulation state."""
        if not self._built:
            self.build()

        # Copy to avoid iterating while updating
        with self._handles_lock:
            entity_handles = list(self._entity_handles.values())
            collision_handles = list(self._collision_entity_handles.values())

        envs_offset = self.envs_offset
        for link_handle in entity_handles:
            link_handle.update(envs_offset)

        for link_handle in collision_handles:
            link_handle.update(envs_offset)

        gui_info = {
            "reset": self._gui_event_state.get("reset_triggered", False),
        }
        self._gui_event_state["reset_triggered"] = False
        return gui_info

    def close(self):
        pass
