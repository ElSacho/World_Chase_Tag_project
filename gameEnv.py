from plateau import Plateau, Case
from cat import Cat
from mouse import Mouse
import pygame
from utils import colors, size
from mouseState import MouseState
from catState import CatState
import random 
import numpy as np

# Define the environment class
class GameEnv:
    def __init__(self, n_cols, n_rows, vision = 2, method = 'speed'):
        self.method = method
        self.n_cols = n_cols
        self.clock = pygame.time.Clock()
        self.n_rows = n_rows
        self.vision = vision
        self.plateau = Plateau(n_cols, n_rows)
        pos = random.sample(range(n_cols * n_rows), 2)
        self.mouse = MouseState(pos[0], self.plateau, vision)
        self.cat = CatState( pos[1], self.plateau, vision)
        # self.mouse = MouseState(0, self.plateau, vision = vision)
        # self.cat = CatState( n_cols* n_rows -1, self.plateau, vision = vision)
        self.cat_observation_space = self.cat.observation_space
        self.cat_action_space = self.cat.action_space
        self.observation_space = self.cat.observation_space
        self.action_space = self.cat.action_space
        self.mouse_observation_space = self.mouse.observation_space
        self.mouse_action_space = self.mouse.action_space
        self.nb_step = 0
        SCREEN_SIZE = (n_cols*size.BLOCK_SIZE, n_rows*size.BLOCK_SIZE)
        self.screen =  pygame.display.set_mode(SCREEN_SIZE)
        #self.screen = pygame.display.set_mode((n_cols*size.BLOCK_SIZE, n_rows*size.BLOCK_SIZE))
        pygame.display.set_caption('Chase Tag')
    
    def reset(self):
        return np.array(self.reset_cat())
        
    def reset_cat(self):
        # pos = random.sample(range(10), 2)
        case_number_ini = random.sample(range(self.n_cols * self.n_rows), 2)
        self.mouse = MouseState(case_number_ini[0], self.plateau, vision = self.vision)
        self.cat = CatState( case_number_ini[1], self.plateau, vision = self.vision)
        return self.cat.get_state(self.mouse)
    
    def reset_mouse(self):
        case_number_ini = random.sample(range(self.n_cols * self.n_rows), 2)
        self.mouse = MouseState(case_number_ini[0], self.plateau, vision = self.vision)
        self.cat = CatState( case_number_ini[1], self.plateau, vision = self.vision)
        return self.mouse.get_state(self.cat)
    
    def get_state_mouse(self):
        return self.mouse.get_state(self.cat)
    
    def draw(self):
        if self.method == 'speed':
            self.plateau.draw_plateau(self.screen) 
            self.cat.draw_cat(self.screen)
            self.mouse.draw_mouse(self.screen)
            pygame.display.update()
        elif self.method == 'human':
            self.plateau.draw_plateau(self.screen) 
            self.cat.draw_cat(self.screen)
            self.mouse.draw_mouse(self.screen)
            pygame.display.update()
            self.clock.tick(size.SPEED)
            

    # Step the environment by taking an action
    def step(self, action):
        return self.cat_step( action)
    
    def cat_step(self, action):
        self.nb_step += 1
        next_state_cat = self.cat.take_action(action, self.mouse)
        done = self.cat.is_done(self.mouse)
        reward = self.cat.get_reward(self.mouse)
        self.cat_state = next_state_cat
        return next_state_cat, reward, done, {}
    
    def mouse_step(self, action):
        next_state_mouse = self.mouse.take_action(action, self.cat)
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