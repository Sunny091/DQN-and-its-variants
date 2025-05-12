# naive_train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from collections import deque
from Gridworld import Gridworld


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_naive_dqn(episodes=300):
    state_dim = 16
    action_dim = 4
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    lr = 1e-3

    policy_net = DQN(state_dim, action_dim)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer = deque(maxlen=10000)
    rewards = []

    for ep in range(episodes):
        env = Gridworld(size=4, mode='static')
        state = env.board.render_np()[..., 0].astype(np.float32).flatten()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(
                        torch.FloatTensor(state).unsqueeze(0))
                    action = q_values.argmax().item()

            env.makeMove(['u', 'd', 'l', 'r'][action])
            reward = env.reward()
            next_state = env.board.render_np(
            )[..., 0].astype(np.float32).flatten()
            done = reward != -1

            buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                states, actions, rewards_, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards_ = torch.FloatTensor(rewards_).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                next_q = policy_net(next_states).max(
                    1, keepdim=True)[0].detach()
                target_q = rewards_ + gamma * (1 - dones) * next_q

                loss = F.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if ep % 10 == 0:
            print(f"Ep {ep}, Reward: {total_reward:.2f}")

    os.makedirs("static/plots", exist_ok=True)
    plt.plot(rewards)
    plt.title("Naive DQN Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("static/plots/naive_dqn_reward.png")
    plt.close()


# Example run
if __name__ == '__main__':
    train_naive_dqn(episodes=300)
