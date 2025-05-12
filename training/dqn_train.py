# training/dqn_train.py
from Gridworld import Gridworld
from models import DQN, DuelingDQN
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import os
from collections import deque
import matplotlib
matplotlib.use('Agg')  # 避免 GUI backend 問題


def train_dqn(model_type='naive', num_episodes=300):
    state_dim = 16  # 4x4 flattened
    action_dim = 4  # up/down/left/right
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'dueling':
        policy_net = DuelingDQN(state_dim, action_dim).to(device)
        target_net = DuelingDQN(state_dim, action_dim).to(device)
    else:
        policy_net = DQN(state_dim, action_dim).to(device)
        target_net = DQN(state_dim, action_dim).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = deque(maxlen=10000)
    rewards = []

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 64

    for ep in range(num_episodes):
        env = Gridworld(size=4, mode='static')  # 用重建方式 reset
        state = env.board.render_np()[..., 0].astype(np.float32).flatten()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q = policy_net(torch.FloatTensor(
                        state).unsqueeze(0).to(device))
                    action = q.argmax().item()

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

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_ = torch.FloatTensor(rewards_).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)

                with torch.no_grad():
                    if model_type == 'double':
                        next_actions = policy_net(
                            next_states).argmax(dim=1, keepdim=True)
                        next_q = target_net(
                            next_states).gather(1, next_actions)
                    else:
                        next_q = target_net(next_states).max(
                            1, keepdim=True)[0]

                    target_q = rewards_ + gamma * (1 - dones) * next_q

                loss = F.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if ep % 10 == 0:
            print(
                f"Ep {ep} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")

        if ep % 50 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # 儲存 JSON 結果
    os.makedirs("data", exist_ok=True)
    with open(f"data/{model_type}_rewards.json", "w") as json_file:
        json.dump(rewards, json_file)

    # 儲存圖表
    os.makedirs("static/plots", exist_ok=True)
    plt.figure()
    plt.plot(rewards)
    plt.title(f"Reward Curve ({model_type.title()} DQN)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(f"static/plots/{model_type}_dqn_reward.png")
    plt.close()

    return rewards


if __name__ == '__main__':
    train_dqn('dueling', 300)
    train_dqn('double', 300)
