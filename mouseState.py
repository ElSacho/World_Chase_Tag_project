from plateau import Plateau
import pygame
from utils import colors, size, distance_between
import numpy as np
import math
from mouse import Mouse
import random
import collections
from plateau import Case


class MouseState(Mouse):
    def __init__(self, starting_position, plateau, vision = 1):
        super().__init__(starting_position, plateau)
        self.vision = vision
        self.position_ini = [(starting_position % plateau.n_cols ), (starting_position // plateau.n_rows)]
        # La vision plus l'ecart de position avec le chasseur
        self.observation_space = (2*vision+1)**2+2
        self.observation_space = (2*vision+1)**2
        self.action_space = 4
        self.action_counter = collections.Counter()
      
      
    def get_value_case(self, case):
        if case.has_cat:
            return -10
        elif case.has_mouse:
            return -3
        elif not case.is_allowed_to_mouse:
            return -1
        elif not case.is_allowed_to_cat:
            return 3
        elif case.timeToSpend == 0:
            return 1
        else : 
            return 1/case.timeToSpend
            
      
    # Get current state of the game  
    def get_state(self, cat):
        self.view = []
        pos = int(self.pos[1]/size.BLOCK_SIZE), int(self.pos[0]/size.BLOCK_SIZE)
        for x in range(pos[0]-self.vision, pos[0]+self.vision+1):
            for y in range(pos[1]-self.vision, pos[1]+self.vision+1):
                if self.pos_isValid([x,y]):
                    case_number = x*self.plateau.n_cols + y
                    self.view.append(self.get_value_case(self.plateau.cases[case_number]))
                else :
                    self.view.append(-1)
        pos_cat = int(cat.pos[1]/size.BLOCK_SIZE), int(cat.pos[0]/size.BLOCK_SIZE)
        # print([self.view[i:i+2*self.vision+1] for i in range(0, (2*self.vision+1)**2, 2*self.vision+1)])
        pos_0 = pos[0]-pos_cat[0]
        pos_1 = pos[1]-pos_cat[1]
        pos_0 = np.sign(pos_0)*max(abs(pos_0), self.vision)
        pos_1 = np.sign(pos_1)*max(abs(pos_1), self.vision)
        # self.view.append(pos_0)
        # self.view.append(pos_1)
        return np.array(self.view)


    def pos_isValid(self, pos):
        if pos[0] >= self.plateau.n_cols or pos[0] < 0:
            return False
        if pos[1] >= self.plateau.n_rows or pos[1] < 0:
            return False
        return True
  
    # Check if the game is finished
    def is_done(self, cat):
        return cat.hasEaten(self)

    # Get the reward for the current state / need to integrate the fact that it can spawn neer the cat 
    def get_reward(self, cat, method = "difference_with_cat_position"):
        if method == 'simple':
            return 1
        elif method == 'with_position':
            pos = int(self.pos[1]/size.BLOCK_SIZE), int(self.pos[0]/size.BLOCK_SIZE)
            pos_cat = int(cat.pos[1]/size.BLOCK_SIZE), int(cat.pos[0]/size.BLOCK_SIZE)
            position = abs(pos[0]-pos_cat[0])+abs(pos[1]-pos_cat[1])
            return math.log(0.1+position)
        elif method == 'difference_with_cat_position':
            previous_distance = distance_between(self.last_pos, cat.pos, self.plateau.n_rows, self.plateau.n_cols)
            actual_distance = distance_between(self.pos, cat.pos, self.plateau.n_rows, self.plateau.n_cols)
            # print(f" last pos : {self.last_pos} and distance : {previous_distance} for {cat.pos}")
            # print(f" actual pos : {self.pos} and distance : {actual_distance} for {cat.pos}")
            if actual_distance > previous_distance:
                return 2
            elif actual_distance == previous_distance: # c'est qu'on est sur un bord
                return -1
            else : return -1
        elif method == 'reward_at_the_end':
            pos = int(self.pos[1]/size.BLOCK_SIZE), int(self.pos[0]/size.BLOCK_SIZE)
            pos_cat = int(cat.pos[1]/size.BLOCK_SIZE), int(cat.pos[0]/size.BLOCK_SIZE)
            distance = abs(pos[0]-pos_cat[0])+abs(pos[1]-pos_cat[1])
            reward = 0
            if distance == 0:
                distance_min = abs(self.position_ini[0]-cat.position_ini[0])+abs(self.position_ini[1]-cat.position_ini[1])
                reward = (self.step - distance_min)/distance_min
            if self.step > (self.plateau.n_cols + self.plateau.n_rows):
                reward = 1
            return reward
        elif method == 'opposite_cat':
            return 1/(0.1+cat.get_reward(self)/100)
    
    def get_actions(self):
            return [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]

    # Take an action and return the next state of the game
    def take_action(self , number_action, cat):
        # action = [0] * 5
        # action[random.randint(0, 4)] = 1
        # number_action = 1
        action = self.get_actions()[number_action]
        self.move(action)
        self.action_counter[number_action] += 1
        return self.get_state(cat)