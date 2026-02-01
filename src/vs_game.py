import pygame
import torch
from src.settings import SCREEN_WIDTH, SCREEN_HEIGHT
from src.game import Game
from src.bird import Bird
from src.food import Food

class VsGame(Game):
    """Simple alternating turns: Player then AI"""
    
    def __init__(self, screen, agent, win_score=10):
        super().__init__(screen)
        self.agent = agent
        self.is_player_turn = True
        self.player_score = 0
        self.ai_score = 0
        self.game_over = False
        self.win_score = win_score  # First to this score wins
        
        # Small fonts to minimize screen clutter
        self.score_font = pygame.font.Font(None, 32)  # Small
        self.turn_font = pygame.font.Font(None, 28)   # Small
        
    def handle_events(self):
        """Player control on player turn"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            
            if self.is_player_turn:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.bird.flap()
                        self.flap_sound.play()
            
            if event.type == self.food_timer:
                self.foods.append(Food(self.food_images))
    
    def update(self):
        """Update with AI or player control"""
        self.bird.update()
        
        # AI control on AI turn
        if not self.is_player_turn:
            self.ai_control()
        
        # Food and collision
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
            self.end_turn()
    
    def ai_control(self):
        """AI makes decisions"""
        state = self.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.agent.model(state_tensor)
        action = torch.argmax(q_values).item()
        
        if action == 1:
            self.bird.flap()
            self.flap_sound.play()
    
    def get_state(self):
        """Get AI state"""
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
    
    def end_turn(self):
        """End current turn, switch players"""
        if self.is_player_turn:
            self.player_score += self.score
            print(f"You scored: {self.score} | Total: {self.player_score}")
        else:
            self.ai_score += self.score
            print(f"AI scored: {self.score} | Total: {self.ai_score}")
        
        # Check if someone won
        if self.player_score >= self.win_score or self.ai_score >= self.win_score:
            self.game_over = True
            self.running = False
            return
        
        # Switch turns
        self.is_player_turn = not self.is_player_turn
        self.score = 0
        self.bird = Bird(300, SCREEN_HEIGHT // 2)
        self.foods = []
        
        self.game_over_sound.play()
        pygame.time.delay(1000)
    
    def draw(self):
        """Draw with minimal text"""
        # Background
        self.bg_x -= 2
        if self.bg_x <= -SCREEN_WIDTH:
            self.bg_x = 0
        self.screen.blit(self.background, (self.bg_x, 0))
        self.screen.blit(self.background, (self.bg_x + SCREEN_WIDTH, 0))
        
        # Game elements
        for food in self.foods:
            food.draw(self.screen)
        self.bird.draw(self.screen)
        
        # Minimal UI - small text, top left only
        score_text = self.score_font.render(f"Round: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (15, 15))
        
        # Turn indicator - very small
        turn_text = "YOU" if self.is_player_turn else "AI"
        turn_color = (50, 200, 50) if self.is_player_turn else (200, 100, 50)
        turn_render = self.turn_font.render(turn_text, True, turn_color)
        self.screen.blit(turn_render, (15, 50))
        
        # Score board - top right, small text
        progress = f"{self.player_score}/{self.win_score} | {self.ai_score}/{self.win_score}"
        board_text = self.turn_font.render(progress, True, (0, 0, 0))
        board_rect = board_text.get_rect(right=SCREEN_WIDTH - 15, top=15)
        self.screen.blit(board_text, board_rect)
        
        pygame.display.update()
