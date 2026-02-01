import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2

        self.model = DQN(self.state_dim, self.action_dim)
        self.target_model = DQN(self.state_dim, self.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        self.memory = []
        self.max_memory_size = 5000

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_decay = 0.999  # slower decay to allow more exploration
        self.epsilon_min = 0.01

        self.batch_size = 64

        self.step_count = 0
        self.target_update_steps = 200  # faster target updates

    def act(self, state):
        # epsilon-greedy with occasional random flip for extra exploration
        if random.random() < self.epsilon or random.random() < 0.05:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.array(states)).float()
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions).squeeze()

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network more frequently
        self.step_count += 1
        if self.step_count % self.target_update_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # epsilon decay per training step
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

