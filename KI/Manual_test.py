import pygame
from Tetris_env import TetrisEnv

def main():
    env = TetrisEnv()
    obs, _ = env.reset()
    done = False
    total_score = 0
    env.skip_render = False

    print("Tastenbelegung:")
    print("← = links | → = rechts | ↓ = runter | ↑ = drehen | Leertaste = harter Drop | ESC = Beenden")

    while not done:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: 
                    done = True
                elif event.key == pygame.K_LEFT:
                    obs, reward, terminated, truncated, _ = env.step(1)
                elif event.key == pygame.K_RIGHT:
                    obs, reward, terminated, truncated, _ = env.step(2)
                elif event.key == pygame.K_DOWN:
                    obs, reward, terminated, truncated, _ = env.step(3)
                elif event.key == pygame.K_UP:
                    obs, reward, terminated, truncated, _ = env.step(4)
                elif event.key == pygame.K_SPACE:
                    obs, reward, terminated, truncated, _ = env.step(5)
                else:
                    continue

                total_score += reward
                print(f"Reward: {reward}")
                done = terminated or truncated

    env.close()
    print(f"Spiel beendet. Gesamtscore: {total_score}")

if __name__ == "__main__":
    main()
