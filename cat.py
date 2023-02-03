from plateau import Plateau
import pygame
from utils import colors, size


class Cat:
    def __init__(self, starting_position, plateau):
        self.case_number = starting_position
        self.plateau = plateau
        self.pos = [self.case_number % plateau.n_cols, self.case_number // plateau.n_rows]
        self.time_to_spend_on_ralentisseur = 0
        self.isOnRalentisseur = False
        
    def move_with_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.move([0,1,0,0,0])
                elif event.key == pygame.K_RIGHT:
                    self.move([0,0,0,1,0])
                elif event.key == pygame.K_LEFT:
                    self.move([0,0,1,0,0])
                elif event.key == pygame.K_DOWN:
                    self.move([1,0,0,0,0])

        
    
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
        ralentisseur = self.analyse_ralentisseur()
        if self.case_number + self.plateau.n_cols < self.plateau.n_cols * self.plateau.n_rows:
            if ralentisseur == 0:
                self.case_number = self.case_number + self.plateau.n_cols
                self.pos = [self.case_number % self.plateau.n_cols, self.case_number // self.plateau.n_rows]
            else :
                lenght_mouvement = int(size.BLOCK_SIZE / (self.plateau.cases[self.case_number].timeToSpend + 1))/size.BLOCK_SIZE 
                self.pos = [self.case_number % self.plateau.n_cols, self.case_number // self.plateau.n_rows + lenght_mouvement]
            
    def move_up(self):
        ralentisseur = self.analyse_ralentisseur()
        if self.case_number - self.plateau.n_cols >= 0:
            if ralentisseur == 0:
                self.case_number = self.case_number - self.plateau.n_cols
                self.pos = [self.case_number % self.plateau.n_cols, self.case_number // self.plateau.n_rows]
            else :
                lenght_mouvement = int(size.BLOCK_SIZE / (self.plateau.cases[self.case_number].timeToSpend + 1))/size.BLOCK_SIZE 
                self.pos = [self.case_number % self.plateau.n_cols, self.case_number // self.plateau.n_rows - lenght_mouvement]
            
            
    def move_right(self):
        ralentisseur = self.analyse_ralentisseur()
        if (self.case_number+1) % self.plateau.n_cols != 0:
            if ralentisseur == 0:
                self.case_number += 1
                self.pos = [self.case_number % self.plateau.n_cols, self.case_number // self.plateau.n_rows]
            else :
                lenght_mouvement = int(size.BLOCK_SIZE / (self.plateau.cases[self.case_number].timeToSpend + 1))/size.BLOCK_SIZE 
                self.pos = [self.case_number % self.plateau.n_cols + lenght_mouvement, self.case_number // self.plateau.n_rows]
            
            
    def move_left(self):
        ralentisseur = self.analyse_ralentisseur()
        if (self.case_number) % self.plateau.n_cols != 0:
            if ralentisseur == 0:
                self.case_number -= 1
                self.pos = [self.case_number % self.plateau.n_cols, self.case_number // self.plateau.n_rows]
            else :
                lenght_mouvement = int(size.BLOCK_SIZE / (self.plateau.cases[self.case_number].timeToSpend + 1))/size.BLOCK_SIZE 
                self.pos = [self.case_number % self.plateau.n_cols - lenght_mouvement, self.case_number // self.plateau.n_rows]
            
    
    def analyse_ralentisseur(self):
        print(self.time_to_spend_on_ralentisseur)
        print(f'case : {self.plateau.cases[self.case_number].timeToSpend}')
        # Si on est sur une case classique, il ne se passe rien, on est pas sur un ralentisseur
        if self.plateau.cases[self.case_number].timeToSpend == 0:
            self.isOnRalentisseur = False
            return 0
        # Sinon si on est sur une case ralentisseur mais qu'on ne doit plus attendre, on avance 
        elif self.time_to_spend_on_ralentisseur == 0 and self.isOnRalentisseur:
            self.isOnRalentisseur = False
            return 0
        # Sinon si on vient d'arriver sur une case ralentisseur, on met a jour le temps d'attente et l etat du chat
        elif self.plateau.cases[self.case_number].timeToSpend !=0 and not self.isOnRalentisseur:
            self.isOnRalentisseur = True
            self.time_to_spend_on_ralentisseur = self.plateau.cases[self.case_number].timeToSpend
            print(f"time : {self.time_to_spend_on_ralentisseur}")
        # Sinon on est juste en attente et on reste la un tempo de plus
        else : 
            self.time_to_spend_on_ralentisseur -= 1
        print(self.time_to_spend_on_ralentisseur)
        # On renvoie le temps d'attente final
        return self.time_to_spend_on_ralentisseur
            
    def hasEaten(self, mouse):
        return self.case_number == mouse.case_number
        
    def draw_cat(self, screen):
        pygame.draw.circle (screen, colors.BLUE1, (self.pos[0]*size.BLOCK_SIZE+size.BLOCK_SIZE/2, self.pos[1]*size.BLOCK_SIZE+size.BLOCK_SIZE/2), size.BLOCK_SIZE/3, 0)
        
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