import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.model(state_tensor)
        action = np.random.choice(len(action_probs[0]), p=action_probs.detach().numpy()[0])
        return action

    def train(self, memory):
        # Implement PPO optimization process using stored experiences in memory
        pass

