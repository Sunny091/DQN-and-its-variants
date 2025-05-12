import torch
import torch.nn as nn


# Naive DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# Dueling DQN
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.feature(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean(dim=1, keepdim=True))
