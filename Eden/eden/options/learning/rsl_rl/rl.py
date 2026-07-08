"""rsl_rl model, distribution, and PPO configuration options."""

from __future__ import annotations

from typing import Any, Literal

from eden.options.learning.rsl_rl.rnd import RslRlRndOptions
from eden.options.learning.rsl_rl.symmetry import RslRlSymmetryOptions
from eden.options.options import ConfigurableOptions
from eden.utils.torch import DeviceStr

####################################
# Distribution configurations      #
####################################


class RslRlDistributionOptions(ConfigurableOptions):
    """Base configuration for an output distribution.

    Parameters
    ----------
    class_name: str
        The distribution class name.
    """

    class_name: str = "GaussianDistribution"


class RslRlGaussianDistributionOptions(RslRlDistributionOptions):
    """Configuration for a state-independent Gaussian distribution.

    Parameters
    ----------
    class_name: str
        The distribution class name.
    init_std: float
        Initial standard deviation.
    std_type: Literal["scalar", "log"]
        Parameterisation of the standard deviation.
    """

    class_name: str = "GaussianDistribution"
    init_std: float = 1.0
    std_type: Literal["scalar", "log"] = "log"


class RslRlHeteroscedasticGaussianDistributionOptions(RslRlDistributionOptions):
    """Configuration for a state-dependent (heteroscedastic) Gaussian distribution.

    The MLP outputs both mean and standard deviation.  The ``init_std``
    value is used to initialise the bias of the std head.

    Parameters
    ----------
    class_name: str
        The distribution class name.
    init_std: float
        Initial standard deviation (used to initialise MLP std-head bias).
    std_type: Literal["scalar", "log"]
        Parameterisation of the standard deviation.
    """

    class_name: str = "HeteroscedasticGaussianDistribution"
    init_std: float = 1.0
    std_type: Literal["scalar", "log"] = "scalar"


##############################
# Model configurations       #
##############################


class RslRlModelOptions(ConfigurableOptions):
    """
    Configuration for an rsl_rl model.

    Parameters
    ----------
    class_name: str
        The rsl_rl class name. This parameter should not be overridden.
    hidden_dims: list[int]
        Hidden layer dimensions for the MLP trunk.
    activation: str
        Activation function.
    obs_normalization: bool
        Whether to normalize observations before the network.
    """

    class_name: str = "MLPModel"
    hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = True


class RslRlModelWithDistributionOptions(RslRlModelOptions):
    """Mixin for models with a distribution.

    Parameters
    ----------
    distribution_cfg: RslRlDistributionOptions | None
        Output distribution.  ``None`` means a deterministic head.
    """

    distribution_cfg: RslRlDistributionOptions | None = RslRlGaussianDistributionOptions()


class RslRlRNNModelOptions(RslRlModelOptions):
    """Configuration for a recurrent (RNN) model."""

    class_name: str = "RNNModel"
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1


class RslRlCNNModelOptions(RslRlModelOptions):
    """Configuration for a CNN-backed model."""

    class_name: str = "CNNModel"
    cnn_cfg: dict[str, dict] | dict[str, Any] | None = None


class RslRlActorRNNOptions(RslRlRNNModelOptions, RslRlModelWithDistributionOptions):
    """Recurrent policy actor."""


class RslRlActorCNNOptions(RslRlRNNModelOptions, RslRlModelWithDistributionOptions):
    """CNN policy actor."""


# Aliases for backward compatibility
RslRlMLPModelOptions = RslRlModelOptions
RslRlActorOptions = RslRlModelWithDistributionOptions
RslRlActorRecurrentOptions = RslRlActorRNNOptions
RslRlCriticOptions = RslRlModelOptions
RslRlCriticRecurrentOptions = RslRlRNNModelOptions
RslRlCriticCNNOptions = RslRlCNNModelOptions

############################
# Algorithm configurations #
############################


class RslRlPpoAlgorithmOptions(ConfigurableOptions):
    """Configuration for the PPO algorithm.

    Default values match the legacy v3.2 ``RslRlPpoAlgorithmOptions``.

    Parameters
    ----------
    class_name: str
        The algorithm class name.
    num_learning_epochs: int
        The number of learning epochs per update.
    num_mini_batches: int
        The number of mini-batches per update.
    learning_rate: float
        The learning rate for the policy.
    schedule: str
        The learning rate schedule.
    gamma: float
        The discount factor.
    lam: float
        The lambda parameter for Generalized Advantage Estimation (GAE).
    entropy_coef: float
        The coefficient for the entropy loss.
    desired_kl: float
        The desired KL divergence.
    max_grad_norm: float
        The maximum gradient norm.
    value_loss_coef: float
        The coefficient for the value loss.
    use_clipped_value_loss: bool
        Whether to use clipped value loss.
    clip_param: float
        The clipping parameter for the policy.
    normalize_advantage_per_mini_batch: bool
        Whether to normalize the advantage per mini-batch.
    share_cnn_encoders: bool
        Whether to share CNN encoders between actor and critic.
    rnd_cfg: RslRlRndOptions | None
        The RND configuration.
    symmetry_cfg: RslRlSymmetryOptions | None
        The symmetry configuration.
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.005
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    normalize_advantage_per_mini_batch: bool = False
    share_cnn_encoders: bool = False
    rnd_cfg: RslRlRndOptions | None = None
    symmetry_cfg: RslRlSymmetryOptions | None = None


#########################
# Runner configurations #
#########################


class RslRlBaseRunnerOptions(ConfigurableOptions):
    """Base configuration of the runner.

    Default values match the legacy v3.2 ``RslRlBaseRunnerOptions``.

    Parameters
    ----------
    seed: int
        The seed for the experiment.
    device: str
        The device for the rl-agent.
    num_steps_per_env: int
        The number of steps per environment per update.
    max_iterations: int
        The maximum number of iterations.
    obs_groups: dict[str, list[str]]
        A mapping from observation sets to observation groups.
        Keys are ``"actor"``, ``"critic"`` (and optionally ``"rnd_state"``).
        Values are lists of observation group names provided by the environment.
    clip_actions: float | None
        The clipping value for actions.
    save_interval: int
        The number of iterations between saves.
    check_for_nan: bool
        Whether to check for NaN values in environment outputs.
    experiment_name: str
        The experiment name.
    run_name: str
        The run name.
    logger: Literal["tensorboard", "neptune", "wandb"]
        The logger to use.
    neptune_project: str
        The neptune project name.
    wandb_project: str
        The wandb project name.
    resume: bool
        Whether to resume a previous training.
    load_run: str
        The run directory to load.
    load_checkpoint: str
        The checkpoint file to load.
    """

    seed: int = 7100
    # "auto" resolves to ``str(gs.device)`` at runner construction time, so the
    # default tracks whatever backend Genesis was initialised with (CUDA, MPS,
    # or CPU). Explicit values (e.g. "cuda:0", "cpu") are passed through unchanged.
    # ``DeviceStr`` rejects typos like "cdua:0" at config load.
    device: DeviceStr = "auto"
    num_steps_per_env: int = 24
    max_iterations: int = 30000
    obs_groups: dict[str, list[str]] = {}
    clip_actions: float | None = None
    save_interval: int = 500
    check_for_nan: bool = True
    experiment_name: str = ""
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "wandb"
    neptune_project: str = "eden"
    wandb_project: str = "eden"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"


class RslRlOnPolicyRunnerOptions(RslRlBaseRunnerOptions):
    """Configuration of the runner for on-policy algorithms.

    The runner carries separate ``actor`` and ``critic`` model configs.

    Parameters
    ----------
    class_name: str
        The runner class name.
    actor: RslRlModelWithDistributionOptions
        The actor model configuration with distribution to sample from.
    critic: RslRlModelOptions
        The critic model configuration.
    algorithm: RslRlPpoAlgorithmOptions
        The algorithm configuration.
    """

    class_name: str = "OnPolicyRunner"
    actor: RslRlModelWithDistributionOptions
    critic: RslRlModelOptions
    algorithm: RslRlPpoAlgorithmOptions
