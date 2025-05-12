# lightning_train.py
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
import random
from collections import deque
from Gridworld import Gridworld


class ReplayBuffer(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class LightningDQN(pl.LightningModule):
    def __init__(self, buffer, state_dim, action_dim, gamma=0.99):
        super().__init__()
        self.model = DQNNet(state_dim, action_dim)
        self.target_model = DQNNet(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = buffer
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()
        self.batch_size = 64
        self.lr = 1e-3
        self.action_dim = action_dim

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        device = self.device
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long,
                               device=device).unsqueeze(1)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32,
                             device=device).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_q = self.target_model(next_states).max(
            1, keepdim=True)[0].detach()
        target_q = rewards + self.gamma * (1 - dones) * next_q
        loss = self.loss_fn(q_values, target_q)
        self.log("train_loss", loss)
        return loss


# Main training loop: generate buffer first, then train once
if __name__ == '__main__':
    buffer = []
    state_dim = 16
    action_dim = 4
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    rewards = []

    for ep in range(300):
        env = Gridworld(size=4, mode='static')
        state = env.board.render_np()[..., 0].astype('float32').flatten()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    model = DQNNet(state_dim, action_dim)
                    q = model(torch.FloatTensor(state).unsqueeze(0))
                    action = q.argmax().item()

            env.makeMove(['u', 'd', 'l', 'r'][action])
            reward = env.reward()
            next_state = env.board.render_np(
            )[..., 0].astype('float32').flatten()
            done = reward != -1

            buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)
        if ep % 10 == 0:
            print(f"Ep {ep}, Reward: {total_reward:.2f}")

    os.makedirs("static/plots", exist_ok=True)
    plt.plot(rewards)
    plt.title("Lightning DQN Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("static/plots/lightning_dqn_reward.png")

    os.makedirs("data", exist_ok=True)
    import json
    with open("data/lightning_rewards.json", "w") as f:
        json.dump(rewards, f)
    plt.close()

    # Start training
    dataset = ReplayBuffer(buffer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LightningDQN(
        buffer=dataset, state_dim=state_dim, action_dim=action_dim)
    trainer = pl.Trainer(max_epochs=5, logger=False,
                         enable_checkpointing=False)
    trainer.fit(model, dataloader)
