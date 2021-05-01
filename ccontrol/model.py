import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """MultiLayer perceptron. Approximates \argmax_a{Q(s,a)}"""

    def __init__(self, state_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

        self.batch_norm = nn.BatchNorm1d(128)

    def forward(self, observations):
        x = F.relu(self.batch_norm(self.fc1(observations)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.tanh(self.fc4(x))


class Critic(nn.Module):
    """Q-network model"""

    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128 + action_size, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.batch_norm = nn.BatchNorm1d(128)

    def forward(self, observations, actions):
        """Get Q(s,a) for all agents."""
        x = F.relu(self.batch_norm(self.fc1(observations)))
        x = torch.cat((x, actions), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
