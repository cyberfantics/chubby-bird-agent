import torch
import time
from src.env import ChubbyBirdEnv
from src.agent import DQNAgent

# --- CONFIG ---
episodes = 25              # total training episodes
max_steps = 8000           # max steps per episode
render_training = True     # True to watch agent, False for fast training

# Initialize environment and agent
env = ChubbyBirdEnv(render=render_training)
agent = DQNAgent()

best_reward = -float("inf")
best_score = -1

# --- TRAINING LOOP ---
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        # store experience and train
        agent.remember((state, action, reward, next_state, done))
        agent.train()

        state = next_state
        total_reward += reward
        steps += 1
        # --- RENDER ---
        if render_training:
            env.render(reward=reward, steps=steps)  # draws the screen
            # optional: slow down for clarity
            # time.sleep(0.01)

        # --- END OF EPISODE ---
        if done or steps >= max_steps:
            # decay epsilon once per episode
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            # save model if improved by score first, otherwise by total reward
            save_reason = None
            if env.score > best_score:
                best_score = env.score
                torch.save(agent.model.state_dict(), "assets/model/best_model.pth")
                save_reason = f"score {env.score}"
            elif total_reward > best_reward and steps >= 2500:
                best_reward = total_reward
                torch.save(agent.model.state_dict(), "assets/model/best_model.pth")
                save_reason = f"reward {total_reward:.2f}"

            if save_reason is not None:
                print(f"✔ Model saved ({save_reason})")

            # log episode stats
            print(
                f"Episode {episode} | "
                f"Steps {steps} | "
                f"Reward {total_reward:.2f} | "
                f"Epsilon {agent.epsilon:.3f}"
            )
            break

# close pygame properly at the end
if render_training:
    import pygame
    pygame.quit()
