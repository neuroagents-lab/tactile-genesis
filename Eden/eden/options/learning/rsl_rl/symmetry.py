"""rsl_rl symmetry-augmentation configuration options."""

from eden.options.options import ConfigurableOptions


class RslRlSymmetryOptions(ConfigurableOptions):
    """Configuration for the symmetry-augmentation in the training.

    When use_data_augmentation is True, the data_augmentation_func is used to generate
    augmented observations and actions. These are then used to train the model.

    When use_mirror_loss is True, the mirror_loss_coeff is used to weight the
    symmetry-mirror loss. This loss is directly added to the agent's loss function.

    If both use_data_augmentation and use_mirror_loss are False, then no symmetry-based
    training is enabled. However, the data_augmentation_func is called to compute and log
    symmetry metrics. This is useful for performing ablations.

    For more information, please check the work from :cite:`mittal2024symmetry`.

    Parameters
    ----------
    use_data_augmentation: bool
        Whether to use symmetry-based data augmentation.
    use_mirror_loss: bool
        Whether to use the symmetry-augmentation loss.
    data_augmentation_func: callable
        The symmetry data augmentation function, with signature
        ``func(env, obs, action) -> (obs, action)`` where:

        - ``env`` (VecEnv): the environment object, used to access its properties.
        - ``obs`` (tensordict.TensorDict | None): the observation dictionary; if None,
          the observation is not used.
        - ``action`` (torch.Tensor | None): the action tensor; if None, the action is
          not used.

        It returns a tuple of the augmented observation dictionary and action tensors,
        either of which can be None if its respective input was None.
    mirror_loss_coeff: float
        The weight for the symmetry-mirror loss.
    """

    use_data_augmentation: bool = False
    use_mirror_loss: bool = False
    # data_augmentation_func: Callable
    mirror_loss_coeff: float = 0.0
