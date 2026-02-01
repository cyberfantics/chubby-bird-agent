# src/food.py
import pygame
import random
from src.settings import SCREEN_WIDTH, SCREEN_HEIGHT, FOOD_SPEED

class Food:
    def __init__(self, images):
        # Pick random image
        self.original_image = random.choice(images)

        # Scale food to reasonable size (e.g., 40x40)
        self.image = pygame.transform.scale(self.original_image, (40, 40))

        # Spawn fully inside the screen vertically
        self.rect = self.image.get_rect(
            midtop=(
                SCREEN_WIDTH + 50,  # Start just outside right edge
                random.randint(10, SCREEN_HEIGHT - 50)  # Stay within screen
            )
        )

    def update(self):
        # Move left to simulate bird flying forward
        self.rect.x -= FOOD_SPEED

    def draw(self, screen):
        screen.blit(self.image, self.rect)
