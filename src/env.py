# src/env.py
import pygame
import numpy as np
from src.settings import *
from src.bird import Bird
from src.food import Food

class ChubbyBirdEnv:
    def __init__(self, render=False):
        self.render_mode = render
        self.screen = None

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.clock = pygame.time.Clock()
        self.food_images = [
            pygame.image.load(f"assets/images/food_{i}.png").convert_alpha()
            for i in range(1, 6)
        ]

        self.reset()

    def reset(self):
        import random
        random_y = random.randint(50, SCREEN_HEIGHT - 100)  # Random Y position
        self.bird = Bird(300, random_y)
        self.foods = []
        self.score = 0
        self.done = False
        self.frame_count = 0

        return self._get_state()

    def step(self, action):
        reward = -0.005  # minimal frame penalty (was killing exploration)
        self.frame_count += 1

        # Bird action
        if action == 1:
            self.bird.flap()

        self.bird.update()

        # Limit bird velocity for stability
        self.bird.velocity = max(min(self.bird.velocity, 15), -15)

        # Prevent bird from going above screen
        if self.bird.rect.top < 0:
            self.bird.rect.top = 0
            self.bird.velocity = 0

        # Spawn food every 60 frames
        if self.frame_count % 60 == 0:
            self.foods.append(Food(self.food_images))

        # Update food and check collisions
        for food in self.foods[:]:
            food.update()

            if self.bird.rect.colliderect(food.rect):
                self.foods.remove(food)
                reward += 10.0  # strong reward for collecting food
                self.score += 1
            elif food.rect.right < 0:
                self.foods.remove(food)
                reward -= 2.0  # stronger penalty for missing food

        # STRONG penalty for being at top - forces exploration
        if self.bird.rect.top < SCREEN_HEIGHT * 0.15:
            reward -= 2.0  # harsh penalty to push away from top
        # MODERATE bonus for being in safe middle region (not too strong)
        elif SCREEN_HEIGHT * 0.25 < self.bird.rect.y < SCREEN_HEIGHT * 0.75:
            reward += 0.05  # small reward for staying in middle (reduced from 0.5)
        # Penalty for being at bottom (near death)
        if self.bird.rect.bottom > SCREEN_HEIGHT * 0.9:
            reward -= 0.5

        # Encourage moving vertically toward nearest food
        if len(self.foods) > 0:
            nearest_food = min(self.foods, key=lambda f: f.rect.x)
            distance = abs(nearest_food.rect.y - self.bird.rect.y)
            
            # Small proximity bonus ONLY if close
            if distance < 100:
                reward += 0.05  # very small bonus for being close

        # Death penalty if hits ground
        if self.bird.rect.bottom > SCREEN_HEIGHT:
            reward -= 10.0
            self.done = True

        state = self._get_state()
        return state, reward, self.done

    def _get_state(self):
        if len(self.foods) == 0:
            food_dx = 1.0
            food_dy = 0.0
        else:
            nearest_food = min(self.foods, key=lambda f: f.rect.x)
            food_dx = (nearest_food.rect.x - self.bird.rect.x) / SCREEN_WIDTH
            food_dy = (nearest_food.rect.y - self.bird.rect.y) / SCREEN_HEIGHT

        bird_y = self.bird.rect.y / SCREEN_HEIGHT
        bird_vel = self.bird.velocity / 10  # normalized velocity

        return np.array([bird_y, bird_vel, food_dx, food_dy], dtype=np.float32)

    def render(self, reward=0, steps=0):
        if not self.render_mode:
            return

        self.screen.fill(BACKGROUND_COLOR)

        # Draw food
        for food in self.foods:
            food.draw(self.screen)

        # Draw bird
        self.bird.draw(self.screen)

        # Display score/reward/steps
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        reward_text = font.render(f"Reward: {reward:.2f}", True, (0, 0, 0))
        steps_text = font.render(f"Steps: {steps}", True, (0, 0, 0))

        self.screen.blit(score_text, (10, 10))
        self.screen.blit(reward_text, (10, 50))
        self.screen.blit(steps_text, (10, 90))

        pygame.display.update()
        self.clock.tick(60)
