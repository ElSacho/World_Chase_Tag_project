from plateau import Plateau, Case
from cat import Cat
from mouse import Mouse
import pygame
from utils import colors, size
from mouseState import MouseState
from catState import CatState

# Define the environment class
class GameEnv:
    def __init__(self, n_cols, n_rows, vision = 2):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.plateau = Plateau(n_cols, n_rows)
        self.mouse = MouseState(0, self.plateau, vision = vision)
        self.cat = CatState( n_cols* n_rows -1, self.plateau, vision = vision)
        self.cat_observation_space = self.cat.observation_space
        self.cat_action_space = self.cat.action_space
        self.mouse_observation_space = self.mouse.observation_space
        self.mouse_action_space = self.mouse.action_space
        self.nb_step = 0
        SCREEN_SIZE = (n_cols*size.BLOCK_SIZE, n_rows*size.BLOCK_SIZE)
        self.screen =  pygame.display.set_mode(SCREEN_SIZE)
        #self.screen = pygame.display.set_mode((n_cols*size.BLOCK_SIZE, n_rows*size.BLOCK_SIZE))
        pygame.display.set_caption('Chase Tag')
    
    # Reset the environment to its initial state
    def reset(self):
        self.mouse = MouseState(0, self.plateau)
        self.cat = CatState( self.n_cols* self.n_rows -1, self.plateau)
        return self.cat.get_state(self.mouse), self.mouse.get_state(self.cat)
    
    def reset_cat(self):
        self.mouse = MouseState(0, self.plateau)
        self.cat = CatState( self.n_cols* self.n_rows -1, self.plateau)
        return self.cat.get_state(self.mouse)
    
    def reset_mouse(self):
        self.mouse = MouseState(0, self.plateau)
        self.cat = CatState( self.n_cols* self.n_rows -1, self.plateau)
        return self.mouse.get_state(self.cat)
    
    def draw(self):
        self.plateau.draw_plateau(self.screen) 
        self.cat.draw_cat(self.screen)
        self.mouse.draw_mouse(self.screen)
        pygame.display.update()

    # Step the environment by taking an action
    def cat_step(self, action):
        self.nb_step += 1
        next_state_cat = self.cat.take_action(self.mouse)
        done = self.cat.is_done(self.mouse)
        reward = self.cat.get_reward(self.mouse)
        self.cat_state = next_state_cat
        return next_state_cat, reward, done, {}
    
    def mouse_step(self, action):
        next_state_mouse = self.mouse.take_action(self.cat)
        done = self.mouse.is_done(self.cat)
        reward = self.mouse.get_reward(self.cat)
        self.mouse_state = next_state_mouse
        return next_state_mouse, reward, done, {}
        
        
# if __name__ == '__main__':
#     game = GameEnv(6, 6)
    
#     #game loop
#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 break

#         next_state, reward, game_over, dic = game.step()
#         game.draw()
#     #   
#         if game_over == True:
#             break     
#     pygame.quit()