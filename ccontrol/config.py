from ccontrol.types import NumberOfSteps


class DDPGConfig:
    # Sampling
    SKIP_FRAMES: int = 1

    # Ornstein-Uhlenbeck noise generator
    ADD_NOISE: bool = True
    MU: float = 0.0
    THETA: float = 0.15
    SIGMA: float = 0.2
    DT: float = 0.5

    # Optimisation
    BUFFER_SIZE: int = 1_000_000
    BATCH_SIZE: int = 128
    DISCOUNT: float = 0.99
    ACTOR_LEARNING_RATE: float = 1e-4
    CRITIC_LEARNING_RATE: float = 1e-3
    ACTOR_WEIGHT_DECAY: float = 0.0
    CRITIC_WEIGHT_DECAY: float = 1e-4
    LEARNING_STARTS: NumberOfSteps = 1000
    TARGET_UPDATE_COEFF: float = 1e-3
    CLIP_TD_ERROR: bool = False
    CLIP_GRADIENTS_ACTOR: bool = False
    CLIP_GRADIENTS_CRITIC: bool = True
    UPDATE_EVERY: NumberOfSteps = 20
    NUM_SGD_ITER: int = 10

    # Logging
    LOG_EVERY: NumberOfSteps = 10

    def __setattr__(self, key, value):
        raise AttributeError("Config objets are immutable")
