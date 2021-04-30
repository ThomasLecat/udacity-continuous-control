from typing import ClassVar, List, Optional

import numpy as np
import torch

from ccontrol.config import DDPGConfig
from ccontrol.environment import SingleAgentEnvWrapper
from ccontrol.model import MultilayerPerceptron
from ccontrol.random_processes import OrnsteinUhlenbeckNoise
from ccontrol.replay_buffer import ReplayBufferInterface, SampleBatch, TorchSampleBatch
from ccontrol.utils import convert_to_torch


class DDPG:
    def __init__(
        self,
        env: SingleAgentEnvWrapper,
        config: ClassVar[DDPGConfig],
        replay_buffer: Optional[ReplayBufferInterface],
    ):
        """DQN agent with the following extensions:
        - [x] Double Q-learning
        - [ ] Dueling Q-learning
        - [ ] Prioritized Experience Replay
        """
        self.env: SingleAgentEnvWrapper = env
        self.replay_buffer: Optional[ReplayBufferInterface] = replay_buffer
        self.config: ClassVar[DDPGConfig] = config

        # Use GPU if available
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Create actor networks
        self.actor = MultilayerPerceptron(
            input_size=env.obs_size,
            hidden_layers=[64, 64, 64],
            output_size=env.num_actions,
        )
        self.target_actor = MultilayerPerceptron(
            input_size=env.obs_size,
            hidden_layers=[64, 64, 64],
            output_size=env.num_actions,
        )
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Create Q networks (critic)
        self.q_network = MultilayerPerceptron(
            input_size=env.obs_size + env.num_actions,
            hidden_layers=[64, 64],
            output_size=1,
        ).to(self.device)
        self.target_q_network = MultilayerPerceptron(
            input_size=env.obs_size + env.num_actions,
            hidden_layers=[64, 64],
            output_size=1,
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Random noise process for exploration
        self.random_process = OrnsteinUhlenbeckNoise(
            size=env.num_actions,
            mu=self.config.MU,
            theta=self.config.THETA,
            sigma=self.config.SIGMA,
        )

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            params=self.actor.parameters(), lr=config.ACTOR_LEARNING_RATE
        )
        self.q_optimizer = torch.optim.Adam(
            params=self.q_network.parameters(), lr=config.CRITIC_LEARNING_RATE
        )

    def compute_action(self, observation: np.ndarray, add_noise: bool) -> np.ndarray:
        with torch.no_grad():
            observation = torch.Tensor(observation).to(self.device)
            # Create fake batch dimension of one | (obs_size) -> (1, obs_size)
            observation = torch.unsqueeze(observation, dim=0)
            # (1, num_actions)
            action = self.actor(observation).squeeze().cpu().numpy()
        if add_noise:
            action += self.random_process.sample()
        if self.config.CLIP_ACTIONS:
            pass
        return action

    def train(self, num_episodes: int) -> List:
        """Train the agent for 'num_episodes' and return the list of undiscounted
        cumulated rewards per episode (sum of rewards over all steps of the episode).
        """
        reward_per_episode: List[float] = []
        num_steps_sampled: int = 0

        for episode_idx in range(1, num_episodes + 1):
            # Log progress
            if episode_idx % self.config.LOG_EVERY == 0:
                window_rewards = reward_per_episode[-self.config.LOG_EVERY :]
                print(
                    f"episode {episode_idx}/{num_episodes}, "
                    f"avg. episode reward: {sum(window_rewards) / len(window_rewards)}, "
                    f"num steps sampled: {num_steps_sampled}"
                )

            # Sample one episode
            observation = self.env.reset()
            episode_length: int = 0
            episode_reward: float = 0.0
            while True:
                action = self.compute_action(observation, self.config.ADD_NOISE)
                next_obs, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(observation, action, reward, done, next_obs)
                observation = next_obs
                episode_length += 1
                episode_reward += reward
                if (
                    episode_length % self.config.UPDATE_EVERY == 0
                    and num_steps_sampled > self.config.LEARNING_STARTS
                ):
                    self.update_once()
                if done is True:
                    break

            reward_per_episode.append(episode_reward)
            num_steps_sampled += episode_length
        return reward_per_episode

    def update_once(self) -> None:
        """Perform one Adam update."""
        sample_batch: SampleBatch = self.replay_buffer.sample(self.config.BATCH_SIZE)
        sample_batch: TorchSampleBatch = convert_to_torch(sample_batch, self.device)
        self.update_critic(sample_batch)
        self.update_actor(sample_batch)
        # Soft update of target Q network
        self.soft_update_target_networks()

    def update_actor(self, sample_batch: TorchSampleBatch) -> None:
        """Perform one Adam update of the actor."""
        self.actor_optimizer.zero_grad()
        # Compute loss
        actions_pred = self.actor(sample_batch.observations)
        actor_loss = -self.q_network(
            torch.cat([sample_batch.observations, actions_pred], dim=1)
        ).mean()
        # Compute and apply gradient
        actor_loss.backward()
        if self.config.CLIP_GRADIENTS:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.actor_optimizer.step()

    def update_critic(self, sample_batch: TorchSampleBatch) -> None:
        """Perform one Adam update on the critic."""
        self.q_optimizer.zero_grad()
        q_loss = self.compute_q_loss(sample_batch)
        q_loss.backward()
        if self.config.CLIP_GRADIENTS:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.q_optimizer.step()

    def compute_q_loss(self, sample_batch: TorchSampleBatch) -> torch.Tensor:
        # Compute TD targets
        # (batch_size, num_actions)
        actions_target_tp1 = self.target_actor(sample_batch.next_observations)
        q_targets_tp1 = self.target_q_network(
            torch.cat([sample_batch.next_observations, actions_target_tp1], dim=1)
        )
        td_targets = (
            sample_batch.rewards
            + (1 - sample_batch.dones) * self.config.DISCOUNT * q_targets_tp1
        ).detach()
        # Compute TD errors
        # (batch_size)
        q_values = self.q_network(
            torch.cat([sample_batch.observations, sample_batch.actions], dim=1)
        )
        td_errors = q_values - td_targets
        if self.config.CLIP_TD_ERROR:
            td_errors = torch.clamp(td_errors, -1, 1)
        return torch.sum(td_errors ** 2)

    def soft_update_target_networks(self) -> None:
        """Update the target networks slowly towards the local networks.

        The magnitude of the update is determined by the config parameter
        TARGET_UPDATE_COEFF, often referred as \tau in papers.
        """
        for local_network, target_network in [
            (self.actor, self.target_actor),
            (self.q_network, self.target_q_network),
        ]:
            target_state_dict = target_network.state_dict()
            for param_name, param_tensor in local_network.state_dict().items():
                target_state_dict[param_name] = (
                    (1 - self.config.TARGET_UPDATE_COEFF)
                    * target_state_dict[param_name]
                    + self.config.TARGET_UPDATE_COEFF * param_tensor
                )
            target_network.load_state_dict(target_state_dict)
