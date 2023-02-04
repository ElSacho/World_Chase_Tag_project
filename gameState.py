from plateau import Plateau, Case
from cat import Cat
from mouse import Mouse
import pygame
from utils import colors, size
from mouseState import MouseState
from catState import CatState

# Define the game state class
class GameState:
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
        
    # Get current state of the game
    def get_state(self):
        return self.cat.get_state(), self.mouse.get_state()

    # Check if the game is finished
    def is_done(self):
        return self.cat.hasEaten(self.mouse)

    # Get the reward for the current state
    def get_reward(self):
        pass

    # Get all possible actions for the current state
    def get_actions(self):
        pass

    # Take an action and return the next state of the game
    def take_action(self, action):
        pass
