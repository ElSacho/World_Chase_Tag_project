from plateau import Plateau, Cases
from cat import Cat
from mouse import Mouse
import pygame

# Define the game state class
class Game:
    def __init__(self, n_cols, n_rows):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.plateau = Plateau(n_cols, n_rows)
        self.mouse = Mouse(0, self.plateau)
        self.cat = Cat( n_cols* n_rows, self.plateau)
        self.screen = pygame.display.set_mode((n_cols*, n_rows*))
        pygame.display.set_caption('Dino')
        
    def step(self):
        self.cat.move_with_keyboard()
        self.mouse.move_with_keyboard()
        self.plateau.draw_plateau(self.screen)
        self.cat.draw_cat(self.screen)
        self.mouse.draw_mouse(self.screen)
        return self.cat.hasEaten(self.mouse)
    
if __name__ == '__main__':
    game = Game()
    
    #game loop
    while True:
        game_over = game.step()
    #    
        if game_over == True:
            break     
    pygame.quit()
        
        
    