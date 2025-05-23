import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from collections import deque
import time

from Tetris_env import TetrisEnv  # Deine eigene Tetris-Umgebung

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "tetris_scores.csv")
model_path = os.path.join(BASE_DIR, "Tetris_DQN.pth")


# DQN (voll verbundenes neuronales Netz)
class DQN(nn.Module):
    def __init__(self, input_dim=200, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def train():
    episodes = int(input("Trainingsepisoden: "))
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TetrisEnv()
    model = DQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer()

    episode_scores = []
    start_episode = 0
    checkpoint_path = os.path.join(BASE_DIR, "Tetris_DQN.pth")

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        episode_scores = df_old["score"].tolist()
        start_episode = len(episode_scores)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint["episode"] + 1
        print(f"Modell geladen, fortsetzen bei Episode {start_episode}")

    for episode in trange(start_episode, start_episode + episodes, desc="Tetris Training"):
        obs, _ = env.reset()
        state = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = q_values.argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_obs.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states)
                next_states = torch.cat(next_states)
                actions = torch.tensor(actions, dtype=torch.long).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    max_next_q_values = model(next_states).max(1)[0]
                    targets = rewards + gamma * max_next_q_values * (1 - dones)

                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        episode_scores.append(total_reward)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    torch.save(model.state_dict(), model_path)
    print("Finales Modell gespeichert als Tetris_DQN.pth")

    df = pd.DataFrame(episode_scores, columns=["score"])
    df.to_csv(csv_path, index=False)
    print("Alle Scores gespeichert in tetris_scores.csv")

    if os.path.exists(csv_path):
        df_all = pd.read_csv(csv_path)
        all_scores = df_all["score"].tolist()

        plt.figure(figsize=(12, 6))
        plt.plot(all_scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Tetris Trainingserfolg (gesamte Historie)")
        plt.grid()
        plt.show()

        print(f"Gesamttraining: {len(all_scores)} Episoden")
        print(f"Durchschnittsscore: {sum(all_scores) / len(all_scores):.2f}")
        print(f"Höchster Score: {max(all_scores)}")
    else:
        print("Warnung: Keine tetris_scores.csv gefunden zum Plotten.")

    # Vorzeigerunde
    obs, _ = env.reset()
    state = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    done = False
    total_reward = 0

    print("\nSpiele eine Vorzeigerunde...")
    fall_interval = 0.5  # alle 0.5 Sekunden automatisch nach unten
    last_fall_time = time.time()

    while not done:
        env.render()
        current_time = time.time()

        if current_time - last_fall_time >= fall_interval:
            action = 3  # runter
            last_fall_time = current_time
        else:
            with torch.no_grad():
                q_values = model(state)
                action = q_values.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = torch.tensor(next_obs.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        total_reward += reward

    print(f"Erspielter Score in der Vorzeigerunde: {total_reward}")
    env.close()



if __name__ == "__main__":
    train()
