"""Mixin adding debug-draw helpers (points, arrows, frames) to environments."""

from __future__ import annotations

import genesis as gs
import numpy as np


class DrawDebugMixin:
    """Mixin providing debug drawing pass-throughs to the Genesis scene."""

    @gs.assert_built
    def draw_debug_line(self, start, end, radius=0.002, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws a line in the scene for visualization.

        Parameters
        ----------
        start : array_like, shape (3,)
            The starting point of the line.
        end : array_like, shape (3,)
            The ending point of the line.
        radius : float, optional
            The radius of the line (represented as a cylinder)
        color : array_like, shape (4,), optional
            The color of the line in RGBA format.

        Returns
        -------
        node : pyrender.mesh.Mesh
            The created debug object.
        """
        return self.scene.draw_debug_line(start, end, radius, color)

    @gs.assert_built
    def draw_debug_arrow(self, pos, vec=(0, 0, 1), radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws an arrow in the scene for visualization.

        Parameters
        ----------
        pos : array_like, shape (3,)
            The starting position of the arrow.
        vec : array_like, shape (3,), optional
            The vector of the arrow.
        radius : float, optional
            The radius of the arrow body (represented as a cylinder).
        color : array_like, shape (4,), optional
            The color of the arrow in RGBA format.

        Returns
        -------
        node : pyrender.mesh.Mesh
            The created debug object.
        """
        return self.scene.draw_debug_arrow(pos, vec, radius, color)

    @gs.assert_built
    def draw_debug_frame(self, T, axis_length=1.0, origin_size=0.015, axis_radius=0.01):
        """
        Draws a 3-axis coordinate frame in the scene for visualization.

        Parameters
        ----------
        T : array_like, shape (4, 4)
            The transformation matrix of the frame.
        axis_length : float, optional
            The length of the axes.
        origin_size : float, optional
            The size of the origin point (represented as a sphere).
        axis_radius : float, optional
            The radius of the axes (represented as cylinders).

        Returns
        -------
        node : pyrender.mesh.Mesh
            The created debug object.
        """
        return self.scene.draw_debug_frame(T, axis_length, origin_size, axis_radius)

    @gs.assert_built
    def draw_debug_frames(self, Ts, axis_length=1.0, origin_size=0.015, axis_radius=0.01):
        """
        Draws 3-axis coordinate frames in the scene for visualization.

        Parameters
        ----------
        Ts : array_like, shape (n, 4, 4)
            The transformation matrices of frames.
        axis_length : float, optional
            The length of the axes.
        origin_size : float, optional
            The size of the origin point (represented as a sphere).
        axis_radius : float, optional
            The radius of the axes (represented as cylinders).

        Returns
        -------
        node : pyrender.mesh.Mesh
            The created debug object.
        """
        return self.scene.draw_debug_frames(Ts, axis_length, origin_size, axis_radius)

    @gs.assert_built
    def draw_debug_mesh(self, mesh, pos=np.zeros(3), T=None):
        """
        Draws a mesh in the scene for visualization.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to be drawn.
        pos : array_like, shape (3,), optional
            The position of the mesh in the scene.
        T : array_like, shape (4, 4) | None, optional
            The transformation matrix of the mesh. If None, the mesh will be drawn at the position specified by `pos`. Otherwise, `T` has a higher priority than `pos`.

        Returns
        -------
        node : pyrender.mesh.Mesh
            The created debug object.
        """
        return self.scene.draw_debug_mesh(mesh, pos, T)

    @gs.assert_built
    def draw_debug_sphere(self, pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws a sphere in the scene for visualization.

        Parameters
        ----------
        pos : array_like, shape (3,)
            The center position of the sphere.
        radius : float, optional
            radius of the sphere.
        color : array_like, shape (4,), optional
            The color of the sphere in RGBA format.

        Returns
        -------
        node : pyrender.mesh.Mesh
            The created debug object.
        """
        return self.scene.draw_debug_sphere(pos, radius, color)

    @gs.assert_built
    def draw_debug_spheres(self, poss, radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws multiple spheres in the scene for visualization.

        Parameters
        ----------
        poss : array_like, shape (N, 3)
            The positions of the spheres.
        radius : float, optional
            The radius of the spheres.
        color : array_like, shape (4,), optional
            The color of the spheres in RGBA format.

        Returns
        -------
        node : pyrender.mesh.Mesh
            The created debug object.
        """
        return self.scene.draw_debug_spheres(poss, radius, color)

    @gs.assert_built
    def draw_debug_box(
        self,
        bounds,
        color=(1.0, 0.0, 0.0, 1.0),
        wireframe=True,
        wireframe_radius=0.0015,
    ):
        """
        Draws a box in the scene for visualization.

        Parameters
        ----------
        bounds : array_like, shape (2, 3)
            The bounds of the box, specified as [[min_x, min_y, min_z], [max_x, max_y, max_z]].
        color : array_like, shape (4,), optional
            The color of the box in RGBA format.
        wireframe : bool, optional
            Whether to draw the box as a wireframe.
        wireframe_radius : float, optional
            The radius of the wireframe lines.

        Returns
        -------
        node : pyrender.mesh.Mesh
            The created debug object.
        """
        return self.scene.draw_debug_box(bounds, color, wireframe=wireframe, wireframe_radius=wireframe_radius)

    @gs.assert_built
    def draw_debug_points(self, poss, colors=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws points in the scene for visualization.

        Parameters
        ----------
        poss : array_like, shape (N, 3)
            The positions of the points.
        colors : array_like, shape (4,), optional
            The color of the points in RGBA format.

        Returns
        -------
        node : pyrender.mesh.Mesh
            The created debug object.
        """
        return self.scene.draw_debug_points(poss, colors)

    @gs.assert_built
    def draw_debug_path(self, qposs, entity, link_idx=-1, density=0.3, frame_scaling=1.0):
        """
        Draws a planned joint trajectory in the scene for visualization.

        Parameters
        ----------
        qposs : array_like, shape (N, M)
            The joint positions of the planned points.
            N is the number of configurations (i.e., trajectory points).
            M is the number of degrees of freedom for the entity (i.e., joint dimensions).
        entity : gs.engine.entities.RigidEntity
            The rigid entity whose forward kinematics are used to compute the trajectory path.
        link_idx : int, optional
            The link id of the rigid entity to visualize. Defeault is -1.
        density : float, optional
            Controls the sampling density of the trajectory points to visualize. Default is 0.3.
        frame_scaling : float, optional
            Scaling factor for the visualization frames' size. Affects the length and thickness of the debug frames.
            Default is 1.0.

        Returns
        -------
        node : pyrender.mesh.Mesh
            The created debug object representing the visualized trajectory.

        Notes
        -----
        The function uses forward kinematics (FK) to convert joint positions to Cartesian space and render debug frames.
        The density parameter reduces FK computational load by sampling fewer points, with 1.0 representing the whole
        trajectory.
        """
        return self.scene.draw_debug_frames(
            qposs=qposs,
            entity=entity,
            link_idx=link_idx,
            density=density,
            frame_scaling=frame_scaling,
        )

    def clear_debug_object(self, node):
        self.scene.clear_debug_object(node)

    def clear_debug_objects(self):
        self.scene.clear_debug_objects()
