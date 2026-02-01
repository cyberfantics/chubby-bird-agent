import pygame
from src.settings import SCREEN_WIDTH, SCREEN_HEIGHT

class SimpleMenu:
    """Clean, simple menu with 3 options"""
    
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.selected = 0
        self.options = ["Manual Play", "AI Play", "You vs AI"]
        self.running = True
        
        # Font
        self.title_font = pygame.font.Font(None, 80)
        self.option_font = pygame.font.Font(None, 50)
        self.small_font = pygame.font.Font(None, 36)
        
        # Colors
        self.bg_color = (135, 206, 235)
        self.selected_color = (255, 200, 0)
        self.normal_color = (255, 255, 255)
        self.text_color = (0, 0, 0)
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.selected = (self.selected - 1) % len(self.options)
                elif event.key == pygame.K_DOWN:
                    self.selected = (self.selected + 1) % len(self.options)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    return self.options[self.selected].lower().replace(" ", "_")
        
        return None
    
    def draw(self):
        self.screen.fill(self.bg_color)
        
        # Title
        title = self.title_font.render("Chubby Bird", True, self.text_color)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 80))
        self.screen.blit(title, title_rect)
        
        # Menu options
        for i, option in enumerate(self.options):
            color = self.selected_color if i == self.selected else self.normal_color
            
            if i == self.selected:
                text = self.option_font.render(f"► {option} ◄", True, color)
            else:
                text = self.option_font.render(option, True, color)
            
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, 300 + i * 90))
            self.screen.blit(text, text_rect)
        
        # Instructions
        inst = self.small_font.render("↑ ↓ to select • ENTER to play", True, self.text_color)
        inst_rect = inst.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 60))
        self.screen.blit(inst, inst_rect)
        
        pygame.display.update()
    
    def run(self):
        while self.running:
            self.clock.tick(60)
            choice = self.handle_events()
            
            if choice == "quit":
                return None
            elif choice:
                return choice
            
            self.draw()
        
        return None
