from plateau import Plateau, Case
from cat import Cat
from mouse import Mouse
import pygame
from utils import colors, size
from mouseState import MouseState, MousesState
from catState import CatState, CatsState
import random 
import numpy as np

# Define the environment class
class GameEnv:
    def __init__(self, n_cols, n_rows, number_of_mouses, number_of_cats, vision = 2, method = 'speed',  method_to_spend_time = "random", cases_to_spend_time = None, method_for_house = "random", case_house = None, method_for_wall = "random", case_wall = None ):
        self.method = method
        self.number_of_mouses = number_of_mouses
        self.number_of_cats = number_of_cats
        self.n_cols = n_cols
        self.clock = pygame.time.Clock()
        self.n_rows = n_rows
        self.vision = vision
        self.plateau = Plateau(n_cols, n_rows,  method_to_spend_time = method_to_spend_time, cases_to_spend_time = cases_to_spend_time, method_for_house = method_for_house, case_house = case_house, method_for_wall = method_for_wall, case_wall = case_wall )
        self.mouses = MousesState(self.plateau, number_of_mouses, vision)
        self.cats = CatsState(self.plateau, number_of_cats, vision)
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
        
    def reset_cats(self):
        # pos = random.sample(range(10), 2)
        for mouse in self.mouses.tabMouses:
            self.plateau.cases[mouse.case_number].has_mouse = False
        for cat in self.cats.tabCats:
            self.plateau.cases[cat.case_number].has_cat = False
        self.mouses = MousesState(self.plateau, self.number_of_mouses, self.vision)
        self.cats = CatsState(self.plateau, self.number_of_cats, self.vision)
    
    def reset_mouses(self):
        for mouse in self.mouses.tabMouses:
            self.plateau.cases[mouse.case_number].has_mouse = False
        for cat in self.cats.tabCats:
            self.plateau.cases[cat.case_number].has_cat = False
        self.mouses = MousesState(self.plateau, self.number_of_mouses, self.vision)
        self.cats = CatsState(self.plateau, self.number_of_cats, self.vision)
    
    def get_state_mouse(self, i):
        return self.mouses.tabMouses[i].get_state(None)
    
    def get_state_cat(self, i):
        return self.cats.tabCats[i].get_state(None)
    
    def draw(self):
        if self.method == 'speed':
            self.plateau.draw_plateau(self.screen) 
            self.cats.draw_cats(self.screen)
            self.mouses.draw_mouses(self.screen)
            pygame.display.update()
        elif self.method == 'human':
            self.plateau.draw_plateau(self.screen) 
            self.cats.draw_cats(self.screen)
            self.mouses.draw_mouses(self.screen)
            pygame.display.update()
            self.clock.tick(size.SPEED)
    
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