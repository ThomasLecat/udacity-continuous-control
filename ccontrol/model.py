import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor Model."""

    def __init__(self, state_size, action_size):
        """Initialize model."""
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, state):
        """Forward pass, get action from state."""
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.tanh(self.fc4(x))


class Critic(nn.Module):
    """Q-network model."""

    def __init__(self, state_size, action_size):
        """Initialize model."""
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128 + action_size, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, observation, action):
        """Forward pass, get Q value from action and state."""
        x = F.relu(self.bn1(self.fc1(observation)))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
