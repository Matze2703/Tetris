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
import pygame
import numpy as np
import torch._dynamo
from gymnasium.vector import SyncVectorEnv

from Tetris_env import TetrisEnv  # Deine eigene Tetris-Umgebung

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "tetris_scores.csv")
model_path = os.path.join(BASE_DIR, "Tetris_DQN.pth")


# DQN (voll verbundenes neuronales Netz)
class DQN(nn.Module):
    def __init__(self, input_dim=2 * 20 * 10, output_dim=6):  # 2 Kanäle mit 20x10
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

pygame.mixer.init()
def play_sound(soundfile):
    sound = pygame.mixer.Sound(f"sound_design\\{soundfile}")
    sound.play()

def train():
    play_sound("line_clear.mp3")
    episodes = int(input("\nTrainingsepisoden: "))
    batch_size = 128     #Belastung von CPU/GPU
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.99
    epsilon_min = 0.05
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_ENVS = 1 # Nur 1 Environment bei CPU
    if torch.cuda.is_available():
        print(f"GPU erkannt: {torch.cuda.get_device_name(0)}")
        NUM_ENVS = 4  # oder 8, je nach Speicher – RTX 3090 schafft locker 8

    def make_env():
        def _init():
            env = TetrisEnv()
            env.skip_render = True  # wichtig!
            return env
        return _init

    envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

    model = DQN().to(device)
    if torch.__version__ >= "2.0":  # Modell komprimieren, um schneller zu trainieren
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, backend="eager")  # kein JIT, aber stabil

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
    else:
        print("Kein Modell gefunden")

    current_phase = -1  # für Logging
    for episode in trange(start_episode, start_episode + episodes, desc="KI Training"):
        # Dynamische Aktionsfreigabe je Episode
        if episode < 1000:
            allowed_actions = [0, 1, 2, 4]  # Phase 1 + Test mit Rotation?
            phase = 0
        elif episode < 2000:
            allowed_actions = [0, 1, 2, 4]  # Phase 2: + Rotation
            phase = 1
        elif episode < 3000:
            allowed_actions = [0, 1, 2, 4, 5]  # Phase 3: + Hard Drop
            phase = 2
        else:
            allowed_actions = [0, 1, 2, 3, 4, 5]  # Phase 4: alles
            phase = 3

        if phase != current_phase:
            print(f"\nAktionsphase {phase} aktiviert (Episode {episode}) - Erlaubt: {allowed_actions}")
            current_phase = phase

        obs, _ = envs.reset()
        state = torch.tensor(obs.reshape(NUM_ENVS, -1), dtype=torch.float32).to(device)


        total_reward = 0
        done = False

        while not done:
            actions = []
            for i in range(NUM_ENVS):
                if random.random() < epsilon:
                    actions.append(random.choice(allowed_actions))
                else:
                    with torch.no_grad():
                        q_values = model(state[i].unsqueeze(0))
                        actions.append(allowed_actions[torch.argmax(q_values[0][allowed_actions]).item()])
            actions = np.array(actions)


            next_obs, rewards, terminated, truncated, _ = envs.step(actions)
            done = terminated or truncated
            next_state = torch.tensor(next_obs.reshape(NUM_ENVS, -1), dtype=torch.float32).to(device)

            for i in range(NUM_ENVS):
                done = terminated[i] or truncated[i]
                replay_buffer.push((state[i].unsqueeze(0),
                                    actions[i],
                                    rewards[i],
                                    next_state[i].unsqueeze(0),
                                    done))

            state = next_state
            total_reward += sum(rewards)

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

    #Speichern des Modells
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": start_episode + episodes - 1
    }, model_path)

    print(f"Finales Modell gespeichert als {model_path}")

    df = pd.DataFrame(episode_scores, columns=["score"])
    df.to_csv(csv_path, index=False)
    print(f"Alle Scores gespeichert in {csv_path}")
    play_sound("line_clear.mp3")
    
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
        print(f"Warnung: Keine {csv_path} gefunden zum Plotten.")

    # Vorzeigerunde(n)
    demo_env = TetrisEnv()
    demo_env.skip_render = False

    while True:
        answer = input("\nVorzeigerunde spielen? (y/n): ").strip().lower()
        if answer == "y":
            play_demo_episode(model, demo_env, device)
        elif answer == "n":
            print(f"Letzter sichtbarer Score: {demo_env.score}")
            demo_env.close()
            break
        else:
            print("Bitte nur 'y' oder 'n' eingeben.")



def play_demo_episode(model, env, device):
    # Ensure rendering is enabled
    env.skip_render = False
    
    obs, _ = env.reset()
    state = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    done = False
    total_reward = 0

    print("\nSpiele eine Vorzeigerunde...")

    while not done:
        env.render()
        
        with torch.no_grad():
            q_values = model(state)
            action = q_values.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = torch.tensor(next_obs.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        total_reward += reward
        print(f"Aktion: {action} | Reward: {reward}")
        time.sleep(0.05)  # Slow down for visibility

    print(f"\n    Vorzeigerunde abgeschlossen:")
    print(f"      Gesamtreward (Trainingssicht): {total_reward}")
    print(f"      Sichtbarer Spiel-Score:        {env.score}")




if __name__ == "__main__":
    train()
