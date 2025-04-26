import random

class BanditAgent:
    def __init__(self, candidate_sizes, token_changes):
        self.arms = [(c, t) for c in candidate_sizes for t in token_changes]
        self.q_values = {arm: 0.0 for arm in self.arms}
        self.counts = {arm: 0 for arm in self.arms}
        self.epsilon = 0.2
    def print_arms(self):
        print(self.q_values)
        
    def select_arm(self):
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        return max(self.arms, key=lambda arm: self.q_values[arm])

    def update(self, arm, reward):
        self.counts[arm] += 1
        alpha = 0.05 #1 / self.counts[arm]
        self.q_values[arm] += alpha * (reward - self.q_values[arm])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent(nn.Module):
    def __init__(self, state_dim=4, action_dim=3, hidden_dim=16, lr=5e-2):
        super(PPOAgent, self).__init__()
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Output action probabilities
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output scalar value
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        action_probs = self.policy_net(state)
        state_value = self.value_net(state)
        return action_probs, state_value

    def act(self, state):
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def update(self, states, actions, rewards, old_logprobs, gamma=0.99, eps_clip=0.2, K_epochs=4):
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Compute discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        for _ in range(K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(states, actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = discounted_rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * (state_values - discounted_rewards) ** 2 - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()