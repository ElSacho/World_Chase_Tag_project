from plateau import Plateau
import pygame
from utils import colors, size


class Cat:
    def __init__(self, starting_position, plateau):
        self.case_number = starting_position
        self.plateau = plateau
        self.pos = [(self.case_number % plateau.n_cols )*size.BLOCK_SIZE + size.BLOCK_SIZE/2, (self.case_number // plateau.n_rows) *size.BLOCK_SIZE + size.BLOCK_SIZE/2]
        self.MouseIsDead = False
        self.step = 0
        
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
        self.step += 1
            
    def update_case_number(self):
        x, y = int(self.pos[1]/size.BLOCK_SIZE), int(self.pos[0]/size.BLOCK_SIZE)
        self.case_number = x*self.plateau.n_cols + y
        if self.plateau.cases[self.case_number].timeToSpend == 0:
            self.pos = [(self.case_number % self.plateau.n_cols )*size.BLOCK_SIZE + size.BLOCK_SIZE/2, (self.case_number // self.plateau.n_rows) *size.BLOCK_SIZE + size.BLOCK_SIZE/2]
        
    def move_down(self):
        if self.case_number + self.plateau.n_cols < self.plateau.n_cols * self.plateau.n_rows:
            lenght_mouvement = 1 / (self.plateau.cases[self.case_number].timeToSpend + 1)
            self.pos[1] += lenght_mouvement * size.BLOCK_SIZE
            self.update_case_number()
            
    def move_up(self):
        if self.case_number - self.plateau.n_cols >= 0:
            lenght_mouvement = 1 / (self.plateau.cases[self.case_number].timeToSpend + 1)
            self.pos[1] -= lenght_mouvement * size.BLOCK_SIZE
            self.update_case_number()
                      
    def move_right(self):
        if (self.case_number+1) % self.plateau.n_cols != 0:
            lenght_mouvement = 1 / (self.plateau.cases[self.case_number].timeToSpend + 1)
            self.pos[0] += lenght_mouvement * size.BLOCK_SIZE
            self.update_case_number()
                     
    def move_left(self):
        if (self.case_number) % self.plateau.n_cols != 0:
            lenght_mouvement = 1 / (self.plateau.cases[self.case_number].timeToSpend + 1)
            self.pos[0] -= lenght_mouvement * size.BLOCK_SIZE
            self.update_case_number()
             
    def hasEaten(self, mouse):
        if self.case_number == mouse.case_number:
            self.MouseIsDead = True
            mouse.isDead = True
            return True
        return False
        
    def draw_cat(self, screen):
        # pygame.draw.circle (screen, colors.BLUE1, (self.pos[0]+size.BLOCK_SIZE/2, self.pos[1]+size.BLOCK_SIZE/2), size.BLOCK_SIZE/3, 0)
        pygame.draw.circle (screen, colors.GREEN, (self.pos[0], self.pos[1]), size.BLOCK_SIZE/3, 0)
