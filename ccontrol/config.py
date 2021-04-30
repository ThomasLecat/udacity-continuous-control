from ccontrol.types import NumberOfSteps


class DDPGConfig:
    # Sampling
    SKIP_FRAMES: int = 1
    CLIP_ACTIONS: bool = True

    # Ornstein-Uhlenbeck noise generator
    ADD_NOISE: bool = True
    MU: float = 0.0
    THETA: float = 0.15
    SIGMA: float = 0.2

    # Optimisation
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    DISCOUNT: float = 0.99
    ACTOR_LEARNING_RATE: float = 0.0005
    CRITIC_LEARNING_RATE: float = 0.0005
    LEARNING_STARTS: NumberOfSteps = 1000
    TARGET_UPDATE_COEFF: float = 0.001
    CLIP_TD_ERROR: bool = False
    UPDATE_EVERY: NumberOfSteps = 4

    # Logging
    LOG_EVERY: NumberOfSteps = 10

    def __setattr__(self, key, value):
        raise AttributeError("Config objets are immutable")
