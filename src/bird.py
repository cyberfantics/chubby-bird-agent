# src/bird.py
import pygame
from src.settings import GRAVITY, FLAP_STRENGTH

class Bird:
    def __init__(self, x, y):
        # Load bird frames for animation
        self.frames = [
            pygame.transform.scale(
                pygame.image.load(f"assets/images/chubby_bird_{i}.png").convert_alpha(),
                (60, 60)
            )
            for i in range(1, 5)
        ]
        self.frame_index = 0
        self.animation_timer = 0
        self.animation_speed = 5  # Update frame every 5 ticks
        self.image = self.frames[self.frame_index]

        # Bird position and physics
        self.rect = self.image.get_rect(center=(x, y))
        self.velocity = 0

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self):
        # Apply gravity
        self.velocity += GRAVITY
        self.rect.y += self.velocity

        # Prevent bird from going above screen
        if self.rect.top < 0:
            self.rect.top = 0
            self.velocity = 0

        # Animate bird
        self.animation_timer += 1
        if self.animation_timer >= self.animation_speed:
            self.animation_timer = 0
            self.frame_index = (self.frame_index + 1) % len(self.frames)
            self.image = self.frames[self.frame_index]

    def draw(self, screen):
        # Optional: tilt bird based on velocity
        tilt = -self.velocity * 3  # adjust for desired tilt effect
        tilted_image = pygame.transform.rotate(self.image, tilt)
        new_rect = tilted_image.get_rect(center=self.rect.center)
        screen.blit(tilted_image, new_rect)
