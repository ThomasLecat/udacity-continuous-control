import argparse

import torch
from unityagents import UnityEnvironment

from ccontrol.agent import DDPG
from ccontrol.config import DDPGConfig
from ccontrol.environment import SingleAgentEnvWrapper
from ccontrol.preprocessors import IdentityPreprocessor
from ccontrol.replay_buffer import UniformReplayBuffer
from ccontrol.utils import write_list_to_csv


def train(environment_path: str, num_episodes: int):
    """Train the agent for 'num_episodes', save the score for each training episode
    and the checkpoint of the trained agent.
    """
    config = DDPGConfig
    preprocessor = IdentityPreprocessor()
    env = UnityEnvironment(environment_path, no_graphics=True)
    env = SingleAgentEnvWrapper(env, preprocessor, skip_frames=config.SKIP_FRAMES)
    replay_buffer = UniformReplayBuffer(config.BUFFER_SIZE)

    agent = DDPG(env=env, config=config, replay_buffer=replay_buffer)
    reward_per_episode = agent.train(num_episodes=num_episodes)
    with open("reward_per_episode.csv", "w") as f:
        write_list_to_csv(f, reward_per_episode)
    with open("ddpg_actor_checkpoint.pt", "wb") as f:
        torch.save(agent.actor, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment_path",
        "-p",
        type=str,
        help="Path to your single agent Unity environment file.",
    )
    parser.add_argument(
        "--num_episodes",
        "-n",
        type=int,
        default=500,
        help="Number of episodes on which to train the agent",
    )
    args = parser.parse_args()
    train(args.environment_path, args.num_episodes)
