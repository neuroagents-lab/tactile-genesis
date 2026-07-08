"""rsl_rl Random Network Distillation (RND) configuration options."""

from eden.options.options import ConfigurableOptions


class RslRlRndWeightScheduleOptions(ConfigurableOptions):
    """Configuration for the weight schedule.

    Parameters
    ----------
    mode: str
        The type of weight schedule.
    """

    mode: str = "constant"


class RslRlRndLinearWeightScheduleOptions(RslRlRndWeightScheduleOptions):
    """Configuration for the linear weight schedule.

    This schedule decays the weight linearly from the initial value to the final value
    between initial_step and before final_step.

    Parameters
    ----------
    mode: str
        The type of weight schedule.
    final_value: float
        The final value of the weight parameter.
    initial_step: int
        The initial step of the weight schedule.
        For steps before this step, the weight is the initial value specified in RslRlRndOptions.weight.
    final_step: int
        The final step of the weight schedule.
        For steps after this step, the weight is the final value specified in final_value.
    """

    mode: str = "linear"
    final_value: float
    initial_step: int
    final_step: int


class RslRlRndStepWeightScheduleOptions(RslRlRndWeightScheduleOptions):
    """Configuration for the step weight schedule.

    This schedule sets the weight to the value specified in final_value at step final_step.

    Parameters
    ----------
    mode: str
        The type of weight schedule.
    final_step: int
        The final step of the weight schedule.
        For steps after this step, the weight is the value specified in final_value.
    final_value: float
        The final value of the weight parameter.
    """

    mode: str = "step"
    final_step: int
    final_value: float


class RslRlRndOptions(ConfigurableOptions):
    """Configuration for the Random Network Distillation (RND) module.

    For more information, please check the work from :cite:`schwarke2023curiosity`.

    Parameters
    ----------
    weight: float
        The weight for the RND reward (also known as intrinsic reward).
        Similar to other reward terms, the RND reward is scaled by this weight.
    weight_schedule: RslRlRndWeightScheduleOptions | None
        The weight schedule for the RND reward. Default is None, which means the weight is constant.
    reward_normalization: bool
        Whether to normalize the RND reward.
    state_normalization: bool
        Whether to normalize the RND state.
    learning_rate: float
        The learning rate for the RND module.
    num_outputs: int
        The number of outputs for the RND module.
    predictor_hidden_dims: list[int]
        The hidden dimensions for the RND predictor network.
        If the list contains -1, then the hidden dimensions are the same as the input dimensions.
    target_hidden_dims: list[int]
        The hidden dimensions for the RND target network.
        If the list contains -1, then the hidden dimensions are the same as the input dimensions.
    """

    weight: float = 0.0
    weight_schedule: RslRlRndWeightScheduleOptions | None = None
    reward_normalization: bool = False
    state_normalization: bool = False
    learning_rate: float = 1e-3
    num_outputs: int = 1
    predictor_hidden_dims: list[int] = [-1]
    target_hidden_dims: list[int] = [-1]
