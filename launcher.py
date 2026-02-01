#!/usr/bin/env python3
"""Simple, clean Chubby Bird launcher"""

import pygame
import torch
import sys
from src.settings import SCREEN_WIDTH, SCREEN_HEIGHT
from src.menu import SimpleMenu
from src.game import Game
from src.agent import DQNAgent
from src.vs_game import VsGame

class AIGame(Game):
    """Simple AI-only game"""
    
    def __init__(self, screen, agent):
        super().__init__(screen)
        self.agent = agent
        self.score_font = pygame.font.Font(None, 48)
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == self.food_timer:
                from src.food import Food
                self.foods.append(Food(self.food_images))
    
    def update(self):
        self.bird.update()
        
        # AI control
        state = self.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.agent.model(state_tensor)
        action = torch.argmax(q_values).item()
        
        if action == 1:
            self.bird.flap()
            self.flap_sound.play()
        
        # Food
        for food in self.foods[:]:
            food.update()
            if self.bird.rect.colliderect(food.rect):
                self.foods.remove(food)
                self.score += 1
                self.eat_sound.play()
            elif food.rect.right < 0:
                self.foods.remove(food)
        
        # Death
        if self.bird.rect.bottom > SCREEN_HEIGHT:
            self.running = False
    
    def get_state(self):
        if len(self.foods) == 0:
            food_dx = 1.0
            food_dy = 0.0
        else:
            nearest_food = min(self.foods, key=lambda f: f.rect.x)
            food_dx = (nearest_food.rect.x - self.bird.rect.x) / SCREEN_WIDTH
            food_dy = (nearest_food.rect.y - self.bird.rect.y) / SCREEN_HEIGHT
        
        bird_y = self.bird.rect.y / SCREEN_HEIGHT
        bird_vel = self.bird.velocity / 10
        
        return [bird_y, bird_vel, food_dx, food_dy]
    
    def draw(self):
        self.bg_x -= 2
        if self.bg_x <= -SCREEN_WIDTH:
            self.bg_x = 0
        self.screen.blit(self.background, (self.bg_x, 0))
        self.screen.blit(self.background, (self.bg_x + SCREEN_WIDTH, 0))
        
        for food in self.foods:
            food.draw(self.screen)
        self.bird.draw(self.screen)
        
        # Minimal text
        score_text = self.score_font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (20, 20))
        
        pygame.display.update()


def main():
    # Initialize pygame ONCE
    pygame.init()
    pygame.mixer.init()
    pygame.font.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Chubby Bird 🐦")
    
    # Main loop
    while True:
        menu = SimpleMenu(screen)
        choice = menu.run()
        
        if choice is None:
            break
        
        # Load agent for AI modes
        agent = None
        if choice in ["ai_play", "you_vs_ai"]:
            agent = DQNAgent()
            try:
                agent.model.load_state_dict(torch.load("assets/model/best_model.pth"))
                agent.model.eval()
                agent.epsilon = 0.0
            except FileNotFoundError:
                print("❌ assets/model/best_model.pth not found! Train first with: python train.py")
                pygame.time.delay(2000)
                continue
        
        # Run selected mode
        if choice == "manual_play":
            pygame.display.set_caption("Chubby Bird 🐦 - Manual")
            game = Game(screen)
            game.run()
            print(f"\nYour Score: {game.score}\n")
        
        elif choice == "ai_play":
            pygame.display.set_caption("Chubby Bird 🐦 - AI")
            game = AIGame(screen, agent)
            game.run()
            print(f"\nAI Score: {game.score}\n")
        
        elif choice == "you_vs_ai":
            pygame.display.set_caption("Chubby Bird 🐦 - You vs AI")
            game = VsGame(screen, agent, win_score=10)  # First to 10 points wins
            game.run()
            print(f"\nFinal: You {game.player_score} - {game.ai_score} AI\n")
            if game.player_score > game.ai_score:
                print("🎉 YOU WIN!\n")
            elif game.ai_score > game.player_score:
                print("🤖 AI WINS!\n")
            else:
                print("🤝 TIE!\n")
    
    pygame.quit()
    print("Thanks for playing! 🐦")


if __name__ == "__main__":
    main()
