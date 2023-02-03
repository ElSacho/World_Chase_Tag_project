from plateau import Plateau
import pygame
from utils import colors, size


class Cat:
    def __init__(self, starting_position, plateau):
        self.case_number = starting_position
        self.plateau = plateau
        self.pos = [self.case_number % plateau.n_cols, self.case_number // plateau.n_rows]
        
    def move_with_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            action = [0,0,0,0,1]
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = [0,1,0,0,0]
                if event.key == pygame.K_RIGHT:
                    action == [0,0,0,1,0]
                if event.key == pygame.K_DOWN:
                    action == [0,0,1,0,0]
                if event.key == pygame.K_DOWN:
                    action == [1,0,0,0,0]
                    
        self.move(action)
        
    
    def move(self, action):
        # ne rien faire
        if action == [0,0,0,0,1]:
            pass
        # aller à droite
        elif action == [0,0,0,1,0]:
            self.move_right()
        # aller à gauche
        elif action == [0,0,1,0,0]:
            self.move_left()
        # aller en haut
        elif action == [0,1,0,0,0]:
            self.move_up()
        # aller en bas
        elif action == [1,0,0,0,0]:
            self.move_down()
            
        
    def move_down(self):
        if self.case_number + self.plateau.n_cols < self.plateau.n_cols * self.plateau.n_rows:
            self.case_number = self.case_number + self.plateau.n_cols
            self.pos = [self.case_number % self.plateau.n_cols, self.case_number // self.plateau.n_rows]
            
    def move_up(self):
        if self.case_number - self.plateau.n_cols >= 0:
            self.case_number = self.case_number - self.plateau.n_cols
            self.pos = [self.case_number % self.plateau.n_cols, self.case_number // self.plateau.n_rows]
            
    def move_right(self):
        if (self.case_number+1) % self.plateau.n_cols != 0:
            self.case_number += 1
            self.pos = [self.case_number % self.plateau.n_cols, self.case_number // self.plateau.n_rows]
            
    def move_left(self):
        if (self.case_number) % self.plateau.n_cols != 0:
            self.case_number -= 1
            self.pos = [self.case_number % self.plateau.n_cols, self.case_number // self.plateau.n_rows]
    
    def hasEaten(self, mouse):
        return self.case_number == mouse.case_number
        
    def draw_cat(self, screen):
        pygame.draw.circle (screen, colors.BLUE1, (self.case_number[0]*size.BLOCK_SIZE+size.BLOCK_SIZE/2, self.case_number[1]*size.BLOCK_SIZE+size.BLOCK_SIZE/2), size.BLOCK_SIZE/3, 0)
        
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