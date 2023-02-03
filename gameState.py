from cgitb import reset
from os import remove
from turtle import width
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('Reinforcement-Learning-for-the-dino-game/arial.ttf', 25)

Point = namedtuple('Point', 'x, y')
# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 25

# Define the game state class
class GameState:
    def __init__(self, nb_state):
        self.nb_state = nb_state
        
    
    # Get current state of the game
    def get_state(self):
        return self.state

    # Check if the game is finished
    def is_done(self):
        pass

    # Get the reward for the current state
    def get_reward(self):
        pass

    # Get all possible actions for the current state
    def get_actions(self):
        pass

    # Take an action and return the next state of the game
    def take_action(self, action):
        pass
