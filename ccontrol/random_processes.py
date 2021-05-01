import numpy as np


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process, used to add noise to the action to increase
    exploration.
    """

    def __init__(self, shape: tuple, mu: float, theta: float, sigma: float, dt: float):
        """Initialize parameters and noise process."""
        self.shape = shape
        self.mu = np.full(shape=shape, fill_value=mu, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt

        self.state = None
        self.reset()

    def reset(self) -> None:
        """Reset the internal state to mu."""
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * np.sqrt(
            self.dt
        ) * np.random.randn(*self.shape).astype(np.float32)
        self.state += dx
        return self.state
