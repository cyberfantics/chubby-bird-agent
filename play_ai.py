import torch
import pygame
from src.settings import SCREEN_WIDTH, SCREEN_HEIGHT
from src.game import Game
from src.agent import DQNAgent

class AIGame(Game):
    """Game class modified to use AI agent instead of keyboard input"""
    
    def __init__(self, screen, agent):
        super().__init__(screen)
        self.agent = agent
        
    def handle_events(self):
        """Override to remove keyboard input and add AI control"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == self.food_timer:
                self.foods.append(__import__('src.food', fromlist=['Food']).Food(self.food_images))
        
        # AI decision: choose action based on current state
        state = self.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.agent.model(state_tensor)
        action = torch.argmax(q_values).item()
        
        # Action 1 = flap
        if action == 1:
            self.bird.flap()
            self.flap_sound.play()
    
    def get_state(self):
        """Get state similar to the training environment"""
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
    
    def game_over(self):
        """Override to show final score"""
        self.game_over_sound.play()
        pygame.time.delay(1000)
        print(f"Game Over! Final Score: {self.score}")
        self.running = False

def main():
    pygame.init()
    pygame.mixer.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Chubby Bird 🐦 - AI Mode")

    # Initialize and load agent
    agent = DQNAgent()
    agent.model.load_state_dict(torch.load("assets/model/best_model.pth"))
    agent.model.eval()
    agent.epsilon = 0.0  # Greedy mode
    
    game = AIGame(screen, agent)
    game.run()

if __name__ == "__main__":
    main()
