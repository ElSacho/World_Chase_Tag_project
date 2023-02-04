from plateau import Plateau, Case
from cat import Cat
from mouse import Mouse
import pygame
from utils import colors, size
from mouseState import MouseState
from catState import CatState

# Define the environment class
class GameEnv:
    def __init__(self, n_cols, n_rows):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.plateau = Plateau(n_cols, n_rows)
        self.mouse = MouseState(0, self.plateau)
        self.cat = CatState( n_cols* n_rows -1, self.plateau)
        self.nb_step = 0
        SCREEN_SIZE = (n_cols*size.BLOCK_SIZE, n_rows*size.BLOCK_SIZE)
        self.screen =  pygame.display.set_mode(SCREEN_SIZE)
        #self.screen = pygame.display.set_mode((n_cols*size.BLOCK_SIZE, n_rows*size.BLOCK_SIZE))
        pygame.display.set_caption('Chase Tag')
    
    # Reset the environment to its initial state
    def reset(self):
        self.mouse = MouseState(0, self.plateau)
        self.cat = CatState( self.n_cols* self.n_rows -1, self.plateau)
        return self.cat.get_state(), self.mouse.get_state()
    
    def draw(self):
        self.plateau.draw_plateau(self.screen) 
        self.cat.draw_cat(self.screen)
        self.mouse.draw_mouse(self.screen)
        pygame.display.update()

    # Step the environment by taking an action
    def step(self):
        if self.nb_step % 2 == 0:
            self.nb_step += 1
            next_state_cat = self.cat.take_action(self.mouse)
            done = self.cat.is_done()
            reward = self.cat.get_reward(self.mouse)
            self.cat_state = next_state_cat
            return next_state_cat, reward, done, {}
        else :
            self.nb_step += 1
            next_state_mouse = self.mouse.take_action(self.cat)
            done = self.mouse.is_done()
            reward = self.mouse.get_reward(self.cat)
            self.mouse_state = next_state_mouse
            return next_state_mouse, reward, done, {}
        
        
if __name__ == '__main__':
    game = GameEnv(10, 10)
    
    #game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        print(game.cat.case_number)
        next_state, reward, game_over, dic = game.step()
        game.draw()
        print(game_over)
    #   
        if game_over == True:
            break     
    pygame.quit()