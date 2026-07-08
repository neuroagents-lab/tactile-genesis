"""ImGui slider overlay for manual DOF teleoperation."""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

import eden as en
from eden.utils.devices.base import DeviceBase

if TYPE_CHECKING:
    from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin
    from genesis.vis.viewer import Viewer


class SliderGUI(DeviceBase):
    """
    ImGui-based slider device for teleoperation.

    Attaches an :class:`~genesis.ext.pyrender.overlay.ImGuiOverlayPlugin` to the Genesis
    viewer and registers a custom side panel with one slider per named DoF plus a Reset
    button. The current slider values are polled via :meth:`get_command`.

    Parameters
    ----------
    viewer: genesis.vis.viewer.Viewer
        The viewer to attach the overlay to, typically ``env.scene.viewer``. The viewer
        must already be built — the plugin is registered immediately.
    dofs_name: list[str]
        Labels for each slider (used as the ImGui label text).
    dofs_pos_limits: torch.Tensor
        Tensor of shape ``(n_dofs, 2)`` with ``[lo, hi]`` for each DoF. ``±inf`` is
        clipped to ``±pi`` to keep sliders bounded.
    reset_callback: Callable[[], None] | None
        Optional callback invoked on the viewer thread when the Reset button is pressed.
    initial_position: torch.Tensor | None
        Initial slider values (also used as the Reset target). If ``None``, defaults to
        zero clipped into ``dofs_pos_limits``.
    plugin: ImGuiOverlayPlugin | None
        Existing overlay plugin to register the panel on. If ``None``, a new plugin is
        created and added to ``viewer``. Pass an existing plugin to share one overlay
        across multiple devices.
    """

    def __init__(
        self,
        viewer: "Viewer",
        dofs_name: list[str],
        dofs_pos_limits: torch.Tensor,
        reset_callback: Callable[[], None] | None = None,
        initial_position: torch.Tensor | None = None,
        plugin: "ImGuiOverlayPlugin | None" = None,
    ):
        from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

        super().__init__()

        self.dofs_name = list(dofs_name)
        limits = dofs_pos_limits.detach().cpu().numpy().astype(np.float32, copy=True)
        limits[limits == -np.inf] = -np.pi
        limits[limits == +np.inf] = +np.pi
        self.dofs_pos_limits = limits

        if initial_position is None:
            defaults = np.zeros(len(self.dofs_name), dtype=np.float32)
        else:
            defaults = initial_position.detach().cpu().numpy().astype(np.float32, copy=True)
        self._defaults = np.clip(defaults, limits[:, 0], limits[:, 1])
        self._readings = self._defaults.copy()
        self._reset_callback = reset_callback
        self._viewer = viewer

        if plugin is None:
            plugin = next(
                (p for p in viewer._viewer_plugins if isinstance(p, ImGuiOverlayPlugin)),
                None,
            )
        if plugin is None:
            plugin = viewer.add_plugin(ImGuiOverlayPlugin())
        self._plugin = plugin
        self._plugin.register_panel(self._draw_panel, section="side")
        en.logger.info("SliderGUI ImGui panel registered.")

    def _draw_panel(self, imgui):
        imgui.text("Teleop Sliders")
        for i, name in enumerate(self.dofs_name):
            lo = float(self.dofs_pos_limits[i, 0])
            hi = float(self.dofs_pos_limits[i, 1])
            changed, new_val = imgui.slider_float(
                f"{name}##teleop_slider_{i}", float(self._readings[i]), lo, hi, "%.3f"
            )
            if changed:
                self._readings[i] = new_val
        if imgui.button("Reset##teleop_slider_reset"):
            self._readings[:] = self._defaults
            if self._reset_callback is not None:
                self._reset_callback()

    def get_raw_data(self) -> torch.Tensor:
        return torch.from_numpy(self._readings.copy()).unsqueeze(0)

    def get_command(self) -> torch.Tensor:
        return torch.from_numpy(self._readings.copy()).unsqueeze(0)

    @property
    def stop_event(self) -> "_ViewerStopShim":
        """Provide a compatibility shim for code polling ``slider.stop_event.is_set()``.

        The ImGui overlay shares the viewer's lifetime, so the teleop loop should
        check ``viewer.is_alive()``. Exposed so existing code that polls
        ``slider.stop_event.is_set()`` keeps working.
        """
        return _ViewerStopShim(self._viewer)

    def close(self) -> None:
        # The ImGui plugin lifecycle is owned by the viewer; nothing to clean up here.
        pass


class _ViewerStopShim:
    def __init__(self, viewer: "Viewer"):
        self._viewer = viewer

    def is_set(self) -> bool:
        return not self._viewer.is_alive()
