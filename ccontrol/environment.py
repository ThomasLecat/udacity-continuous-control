from abc import abstractmethod
from typing import Tuple

import numpy as np
from unityagents import UnityEnvironment

from ccontrol.preprocessors import PreprocessorInterface


class GenericEnvWrapper:
    def __init__(
        self,
        env: UnityEnvironment,
        preprocessor: PreprocessorInterface,
        skip_frames: int,
    ):
        self.env = env
        self.preprocessor = preprocessor
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        assert skip_frames > 0
        self.skip_frames = skip_frames

        self._num_actions = self.brain.vector_action_space_size
        self._obs_size = self.preprocessor.observation_size(
            raw_obs_size=self.brain.vector_observation_space_size
        )

    @abstractmethod
    def reset(self):
        raise NotImplementedError(
            "GenericEnvWrapper is an abstract class and should not be instantiated"
        )

    @property
    def action_space(self) -> Tuple:
        raise NotImplementedError(
            "GenericEnvWrapper is an abstract class and should not be instantiated"
        )

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def obs_size(self) -> int:
        return self._obs_size


class SingleAgentEnvWrapper(GenericEnvWrapper):
    """Wrap the Unity environment into a format similar to single agents gym
    environments.

    Call the preprocessor on observations before sending them back to the agent.
    """

    def reset(self) -> np.ndarray:
        env_info = self.env.reset()[self.brain_name]
        return self.preprocessor.transform(
            env_info.vector_observations[0].astype(np.float32)
        )

    def step(self, action: np.ndarray) -> Tuple:
        for _ in range(self.skip_frames):
            env_info = self.env.step(action)[self.brain_name]
        assert (
            len(env_info.rewards)
            == len(env_info.local_done)
            == env_info.vector_observations.shape[0]
            == 1
        ), "More than one agent found for this environment."
        return (
            self.preprocessor.transform(
                env_info.vector_observations[0].astype(np.float32)
            ),
            env_info.rewards[0],
            env_info.local_done[0],
            None,
        )


class MultiAgentEnvWrapper(GenericEnvWrapper):
    def __init__(
        self,
        env: UnityEnvironment,
        preprocessor: PreprocessorInterface,
        skip_frames: int,
    ):
        super().__init__(env, preprocessor, skip_frames)

        self.__action_space = self.env.reset()[
            self.brain_name
        ].previous_vector_actions.shape

    def reset(self):
        env_info = self.env.reset()[self.brain_name]
        return self.preprocessor.transform(
            env_info.vector_observations.astype(np.float32)
        )

    def step(self, action: np.ndarray) -> Tuple:
        for _ in range(self.skip_frames):
            env_info = self.env.step(action)[self.brain_name]
        return (
            self.preprocessor.transform(
                env_info.vector_observations.astype(np.float32)
            ),
            env_info.rewards,
            env_info.local_done,
            None,
        )

    @property
    def action_space(self) -> Tuple:
        return self.__action_space

    @property
    def num_agents(self) -> int:
        return self.__action_space[0]
