from plateau import Plateau
import pygame
from utils import colors, size
import numpy as np
import math
from cat import Cat
import random


class CatState(Cat):
    def __init__(self, starting_position, plateau, vision):
        super().__init__(starting_position, plateau)
        self.vision = vision
        self.position_ini = [(starting_position % plateau.n_cols ), (starting_position // plateau.n_rows)]
        # La vision plus l'ecart de position avec le chasseur
        self.observation_space = (2*vision+1)**2+2
        self.observation_space = (2*vision+1)**2
        self.action_space = 4
            
    def get_value_case(self, case):
        if case.has_mouse:
            return 10
        if case.has_cat:
            return -3
        elif not case.is_allowed_to_cat and case.is_allowed_to_mouse:
            return 5
        elif not case.is_allowed_to_cat:
            return -1
        elif case.timeToSpend == 0:
            return 1
        else : 
            return 1/case.timeToSpend 
        

    # Get current state of the game  
    def get_state(self, mouse):
        self.view = []
        pos = int(self.pos[1]/size.BLOCK_SIZE), int(self.pos[0]/size.BLOCK_SIZE)
        for x in range(pos[0]-self.vision, pos[0]+self.vision+1):
            for y in range(pos[1]-self.vision, pos[1]+self.vision+1):
                if self.pos_isValid([x,y]):
                    case_number = x*self.plateau.n_cols + y
                    self.view.append(self.get_value_case(self.plateau.cases[case_number]))
                else :
                    self.view.append(-1)
        # pos_mouse = int(mouse.pos[1]/size.BLOCK_SIZE), int(mouse.pos[0]/size.BLOCK_SIZE)
        # # print([self.view[i:i+2*self.vision+1] for i in range(0, (2*self.vision+1)**2, 2*self.vision+1)])
        # pos_0 = pos[0]-pos_mouse[0]
        # pos_1 = pos[1]-pos_mouse[1]
        # pos_0 = np.sign(pos_0)*max(abs(pos_0), self.vision)
        # pos_1 = np.sign(pos_1)*max(abs(pos_1), self.vision)
        # # self.view.append(pos_0)
        # # self.view.append(pos_1)
        return np.array(self.view)
                     
    def pos_isValid(self, pos):
        if pos[0] >= self.plateau.n_cols or pos[0] < 0:
            return False
        if pos[1] >= self.plateau.n_rows or pos[1] < 0:
            return False
        return True
  
    # Check if the game is finished
    def is_done(self, mouse):
        return self.hasEaten(mouse) or self.step > 5*(self.plateau.n_cols * self.plateau.n_rows)
    
    # Get the reward for the current state
    def get_reward(self, mouse):
        pos = int(self.pos[1]/size.BLOCK_SIZE), int(self.pos[0]/size.BLOCK_SIZE)
        pos_mouse = int(mouse.pos[1]/size.BLOCK_SIZE), int(mouse.pos[0]/size.BLOCK_SIZE)
        distance = abs(pos[0]-pos_mouse[0])+abs(pos[1]-pos_mouse[1])
        reward = 0
        if distance == 0:
            distance_min = abs(self.position_ini[0]-mouse.position_ini[0])+abs(self.position_ini[1]-mouse.position_ini[1])
            reward = 100 / self.step * distance_min
        if self.step > 4*(self.plateau.n_cols + self.plateau.n_rows) == 0:
            reward = -1
        return reward

    # Get all possible actions for the current state
    def get_actions(self):
        return [ [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]]

    # Take an action and return the next state of the game
    def take_action(self , number_action, mouse):
        # action = [0] * 5
        # action[random.randint(0, 4)] = 1        
        action = self.get_actions()[number_action]
        self.move(action)
        return self.get_state(mouse)


class CatsState:
    def __init__(self, plateau, number_of_cats, vision):
        self.tabCats = []
        self.plateau = plateau
        for _ in range(number_of_cats):
            self.tabCats.append(CatState(0, self.plateau, vision))
            
    def drawCats(self, screen):
        for cat in self.tabCats:
            cat.draw_cat(screen)