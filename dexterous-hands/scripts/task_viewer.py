"""
Task environment viewer and debugger.
Provides a GUI to manually control actions and step through the simulation.

Usage:
    python task_viewer.py --task in_hand_repose
"""

import time
from collections import deque
from typing import Optional

import eden as en
import genesis as gs
import numpy as np
import torch
import viser.uplot as uplot
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.extensions.visualization.viser import ViserViewer

from registry import get_argparser, get_task_config_from_args, make_runner


class TaskViewerViser(ViserViewer):
    def __init__(
        self,
        env,
        num_actions: int,
        policy=None,
        control_env=None,
        sample_policy_actions: bool = False,
        host: str = "localhost",
        port: int = 8080,
        enable_gui: bool = True,
    ):
        super().__init__(env, host=host, port=port, enable_gui=enable_gui)
        self.control_env = control_env or env
        self.num_actions = num_actions
        self.policy = policy
        self.sample_policy_actions = sample_policy_actions
        self.actions = torch.zeros(env.num_envs, num_actions, device=self.control_env.device)
        self.use_policy_actions = False
        self.auto_step = False
        self.auto_step_realtime = True
        self.env_step_wall_dt = float(env.dt)
        self._last_auto_step_wall_time: float | None = None
        self.pending_step = False
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_obs = None
        self._action_sliders: list = []
        self._status_text = None

        self.max_history = 1000
        self._reward_plot_names = self._reward_series_labels()
        self._num_reward_series = len(self._reward_plot_names)
        self.time_history = deque(maxlen=self.max_history)
        self.reward_history_per_series = [deque(maxlen=self.max_history) for _ in range(self._num_reward_series)]
        self.start_time = time.time()
        self._reward_plot_handle = None

    @staticmethod
    def _format_vec3(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{float(v):.6f}" for v in values) + "]"

    def _get_active_client_camera(self):
        active_client = None
        active_timestamp = float("-inf")
        for client in self.server.get_clients().values():
            try:
                timestamp = client.camera.update_timestamp
            except AssertionError:
                continue
            if timestamp > active_timestamp:
                active_timestamp = timestamp
                active_client = client
        return active_client.camera if active_client is not None else None

    def print_current_camera_yaml(self) -> None:
        camera = self._get_active_client_camera()
        if camera is None:
            print("Viewer camera: no connected client with camera state yet.")
            return

        fov_degrees = float(np.degrees(camera.fov))
        print("Viewer camera (paste under cameras_options.rec):")
        print(f"  cam_pos: {self._format_vec3(camera.position)}")
        print(f"  cam_lookat: {self._format_vec3(camera.look_at)}")
        print(f"  cam_up: {self._format_vec3(camera.up_direction)}")
        print(f"  cam_fov: {fov_degrees:.6f}")

    def _reward_series_labels(self) -> list[str]:
        rm = getattr(self.env, "reward_manager", None)
        if rm is not None and rm.active_terms:
            return list(rm.active_terms)
        return ["total"]

    def _setup_reward_plot(self) -> None:
        data = (
            np.array([0.0]),
            *[np.array([0.0]) for _ in range(self._num_reward_series)],
        )
        colors = ["red", "green", "blue", "orange", "purple", "cyan", "magenta", "yellow"]
        self._reward_plot_handle = self.server.gui.add_uplot(
            data=data,
            series=(
                uplot.Series(label="Time (s)"),
                *[
                    uplot.Series(
                        label=self._reward_plot_names[i],
                        stroke=colors[i % len(colors)],
                        width=2,
                    )
                    for i in range(self._num_reward_series)
                ],
            ),
            title="Rewards (env 0)",
            scales={
                "x": uplot.Scale(time=False, auto=True),
                "y": uplot.Scale(auto=True),
            },
            legend=uplot.Legend(show=True),
            aspect=2.5,
        )

    def _clear_reward_plot_history(self) -> None:
        self.time_history.clear()
        for dq in self.reward_history_per_series:
            dq.clear()
        self.start_time = time.time()
        if self._reward_plot_handle is not None:
            self._reward_plot_handle.data = (
                np.array([0.0]),
                *[np.array([0.0]) for _ in range(self._num_reward_series)],
            )

    def _append_reward_plot_sample(self, rewards: torch.Tensor) -> None:
        if self._reward_plot_handle is None:
            return
        dt = float(self.env.dt)
        current_time = time.time() - self.start_time
        self.time_history.append(current_time)

        rm = getattr(self.env, "reward_manager", None)
        if rm is not None and rm.active_terms:
            step_contrib = rm._step_reward[0] * dt
            for i in range(self._num_reward_series):
                self.reward_history_per_series[i].append(float(step_contrib[i].item()))
        else:
            self.reward_history_per_series[0].append(float(rewards[0].item()))

        self._reward_plot_handle.data = (
            np.array(list(self.time_history)),
            *[np.array(list(self.reward_history_per_series[i])) for i in range(self._num_reward_series)],
        )

    def _setup_gui(self):
        super()._setup_gui()

        with self.server.gui.add_folder("Control"):
            btn_step = self.server.gui.add_button("Single Step")
            btn_reset = self.server.gui.add_button("Reset Env")
            cb_auto = self.server.gui.add_checkbox("Auto Step", initial_value=False)
            cb_auto_realtime = self.server.gui.add_checkbox(
                "Clip autostep FPS to realtime",
                initial_value=True,
            )
            cb_policy = self.server.gui.add_checkbox("Use Policy Actions", initial_value=False)
            cb_sample_policy = self.server.gui.add_checkbox(
                "Sample Policy Actions",
                initial_value=self.sample_policy_actions,
            )
            btn_zero = self.server.gui.add_button("Zero Actions")
            btn_random = self.server.gui.add_button("Randomize Actions")
            btn_print_camera = self.server.gui.add_button("Print Camera YAML")
            self._status_text = self.server.gui.add_text(
                "Info for env=0",
                initial_value="step=0, ep_rew=0.000",
            )

            @btn_step.on_click
            def _(_) -> None:
                self.pending_step = True

            @btn_reset.on_click
            def _(_) -> None:
                self.reset_env()

            @cb_auto.on_update
            def _(_) -> None:
                self.auto_step = cb_auto.value
                self._last_auto_step_wall_time = None

            @cb_auto_realtime.on_update
            def _(_) -> None:
                self.auto_step_realtime = cb_auto_realtime.value
                self._last_auto_step_wall_time = None

            @cb_policy.on_update
            def _(_) -> None:
                if cb_policy.value and self.policy is None:
                    cb_policy.value = False
                self.use_policy_actions = cb_policy.value
                self._set_action_sliders_disabled(self.use_policy_actions)

            @cb_sample_policy.on_update
            def _(_) -> None:
                self.sample_policy_actions = cb_sample_policy.value

            @btn_zero.on_click
            def _(_) -> None:
                self.zero_actions()

            @btn_random.on_click
            def _(_) -> None:
                self.randomize_actions()

            @btn_print_camera.on_click
            def _(_) -> None:
                self.print_current_camera_yaml()

        with self.server.gui.add_folder("Actions"):
            for i in range(self.num_actions):
                slider = self.server.gui.add_slider(
                    f"Action {i}",
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    initial_value=0.0,
                )

                @slider.on_update
                def _(_, idx=i, handle=slider) -> None:
                    self.update_action(idx, handle.value)

                self._action_sliders.append(slider)

        self._setup_reward_plot()

    def _set_action_sliders_disabled(self, disabled: bool) -> None:
        for slider in self._action_sliders:
            slider.disabled = disabled

    def update_action(self, idx: int, val: float) -> None:
        self.actions[:, idx] = float(val)

    def zero_actions(self) -> None:
        for i, slider in enumerate(self._action_sliders):
            slider.value = 0.0
            self.actions[:, i] = 0.0

    def randomize_actions(self) -> None:
        for i, slider in enumerate(self._action_sliders):
            val = float(np.random.uniform(-1.0, 1.0))
            slider.value = val
            self.actions[:, i] = val

    def _compute_actions(self, obs: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_policy_actions and self.policy is not None and obs is not None:
            with torch.no_grad():
                actions = self.policy(obs, stochastic_output=True) if self.sample_policy_actions else self.policy(obs)
            self.actions[:] = actions
            if self._action_sliders:
                for i, slider in enumerate(self._action_sliders):
                    slider.value = float(actions[0, i].item())
            return actions
        return self.actions

    def step_env(self) -> None:
        output = self.control_env.step(self._compute_actions(self.last_obs))
        if len(output) == 4:
            obs, rewards, dones, extras = output
        else:
            obs, rewards, terminated, truncated, extras = output
            dones = terminated | truncated
        self.last_obs = obs

        self.step_count += 1
        self.episode_reward += rewards[0].item()

        self._append_reward_plot_sample(rewards)
        self.update_info_display()

        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset(dones)

        if dones[0]:
            self.step_count = 0
            self.episode_reward = 0.0
            self.update_info_display()

    def reset_env(self) -> None:
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()
        output = self.control_env.reset()
        self.last_obs = output[0] if isinstance(output, (tuple, list)) else output
        self.step_count = 0
        self.episode_reward = 0.0
        self._clear_reward_plot_history()
        self.update_info_display()

    def update_info_display(self) -> None:
        if self._status_text is not None:
            self._status_text.value = f"steps: {self.step_count}, ep_rew: {self.episode_reward:.3f}"

    def update(self):
        should_step = False
        if self.pending_step:
            should_step = True
            self.pending_step = False
        elif self.auto_step:
            if self.auto_step_realtime:
                current_time = time.perf_counter()
                if self._last_auto_step_wall_time is None:
                    self._last_auto_step_wall_time = current_time
                    should_step = True
                elif (current_time - self._last_auto_step_wall_time) >= self.env_step_wall_dt:
                    self._last_auto_step_wall_time = current_time
                    should_step = True
            else:
                should_step = True

        if should_step:
            self.step_env()

        super().update()


def main():
    """Main function for task viewer."""

    parser = get_argparser(description="View and debug task environments with manual control.")
    parser.add_argument("--envs", "-b", type=int, default=None, help="Number of environments to show")
    parser.add_argument("--viewer", "-v", action="store_true", help="Use Genesis viewer instead of Viser web viewer")
    parser.add_argument("--stochastic", action="store_true", help="Sample from policy distribution")
    args = parser.parse_args()

    # Initialize Eden
    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path="logs/temp",
    )

    # Load task config
    config = get_task_config_from_args(args, upload_logs=False)
    if args.envs is not None:
        config.env_options.num_eval_envs = args.envs
    config.env_options.background_color = (1.0, 1.0, 1.0)

    # Create environment
    env = RslRlVecEnvWrapper.from_config(config, show_viewer=args.viewer, eval_mode=True)

    # Load policy from checkpoint (optional)
    policy = None
    if args.checkpoint or args.actor_checkpoint:
        runner = make_runner(
            env,
            checkpoint=args.checkpoint,
            actor_checkpoint=args.actor_checkpoint,
            critic_checkpoint=args.critic_checkpoint,
        )
        policy = runner.get_inference_policy(device=gs.device)

    is_running = True

    if args.viewer:
        zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        use_policy = policy is not None
        if use_policy:
            print(f"Using policy actions from checkpoint (stochastic={args.stochastic}).")
        else:
            print("No checkpoint loaded; stepping with zero actions.")

        num_steps = 1
        try:
            while is_running:
                for step in range(num_steps):
                    if use_policy:
                        with torch.no_grad():
                            actions = policy(obs, stochastic_output=True) if args.stochastic else policy(obs)
                    else:
                        actions = zero_actions
                    output = env.step(actions)
                    obs, rewards, dones, extras, *_ = output
                    if use_policy and hasattr(policy, "reset"):
                        policy.reset(dones)
                    print(f"Step {step + 1}/{num_steps}")
                    print(f"  Reward: {rewards[0].item()}")
                    print(f"  Done: {dones[0]}")
                    print(f"  Extras: {extras}")
                    print("--------------------------------")

                user_input = input("Num steps to step (q to quit): ")
                if len(user_input) > 0 and user_input[0] == "q":
                    is_running = False
                elif user_input == "debug":
                    import IPython

                    IPython.embed()
                else:
                    try:
                        num_steps = int(user_input)
                    except ValueError:
                        print(f"Running {num_steps} steps...")
        except KeyboardInterrupt:
            print("\nStopping...")
    else:
        viewer = TaskViewerViser(
            env.unwrapped,
            env.num_actions,
            policy=policy,
            control_env=env,
            sample_policy_actions=args.stochastic,
        )
        viewer.build()
        viewer.reset_env()

        try:
            while True:
                viewer.update()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            viewer.close()


if __name__ == "__main__":
    main()
