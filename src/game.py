# src/game.py
import pygame
import random
from src.settings import *
from src.bird import Bird
from src.food import Food

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True
        self.bg_x = 0
        self.score = 0

        # Load background
        self.background = pygame.image.load(
            "assets/images/background_resized.png"
        ).convert()

        # Load food images (scaled inside Food class)
        self.food_images = [
            pygame.image.load(f"assets/images/food_{i}.png").convert_alpha()
            for i in range(1, 6)
        ]

        # Load sounds
        self.flap_sound = pygame.mixer.Sound("assets/sounds/flap.wav")
        self.eat_sound = pygame.mixer.Sound("assets/sounds/eat.wav")
        self.game_over_sound = pygame.mixer.Sound("assets/sounds/game_over.wav")

        pygame.mixer.music.load("assets/sounds/background_music.mp3")
        pygame.mixer.music.play(-1)

        # Create animated bird (x fixed)
        self.bird = Bird(300, SCREEN_HEIGHT // 2)

        # Food list
        self.foods = []

        # Food spawn timer
        self.food_timer = pygame.USEREVENT + 1
        pygame.time.set_timer(self.food_timer, FOOD_SPAWN_TIME)

        # Ensure font module is initialized
        try:
            pygame.font.init()
        except:
            pass
        
        # Font for score with fallbacks
        try:
            self.font = pygame.font.SysFont("arial", 36)
        except:
            try:
                self.font = pygame.font.Font(None, 36)
            except:
                self.font = pygame.font.Font(None, 24)

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()

        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.bird.flap()
                    self.flap_sound.play()

            if event.type == self.food_timer:
                self.foods.append(Food(self.food_images))

    def update(self):
        self.bird.update()

        for food in self.foods[:]:
            food.update()

            if self.bird.rect.colliderect(food.rect):
                self.foods.remove(food)
                self.score += 1
                self.eat_sound.play()
            elif food.rect.right < 0:
                self.foods.remove(food)

        # Game over only when hitting bottom
        if self.bird.rect.bottom > SCREEN_HEIGHT:
            self.game_over()

    def draw(self):
        # Scroll background
        self.bg_x -= 2
        if self.bg_x <= -SCREEN_WIDTH:
            self.bg_x = 0
        self.screen.blit(self.background, (self.bg_x, 0))
        self.screen.blit(self.background, (self.bg_x + SCREEN_WIDTH, 0))

        # Draw food (behind bird)
        for food in self.foods:
            food.draw(self.screen)

        # Draw animated bird on top
        self.bird.draw(self.screen)

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))

        pygame.display.update()

    def game_over(self):
        self.game_over_sound.play()
        pygame.time.delay(1000)
        self.running = False
