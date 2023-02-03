import pygame
from utils import colors, size

class Case:
    def __init__(self, case_number, n_cols, n_rows):
        # numero de 0 à n_cols*n_rows - 1
        self.case_number = case_number
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.speed = 1
        self.pos = [case_number % n_cols, case_number // n_rows]
        
        # self.right = 
        # self.left = 
        # self.up = 
        # self.down = 
        
    def add_velocity(self, speed = 0.5):
        self.speed = speed
        
    def draw_case(self, screen):
        # Dessiner le carré dont la couleur dépend de la vélocité avec un contour noir
        pygame.draw.rect(screen, colors.BLACK, (self.pos[0]*size.BLOCK_SIZE, self.pos[1]*size.BLOCK_SIZE, size.BLOCK_SIZE, size.BLOCK_SIZE), 0)
        
        if self.speed == 1:
            pygame.draw.rect(screen, colors.WHITE, (self.pos[0]*size.BLOCK_SIZE+size.CONTOUR_SIZE, self.pos[1]*size.BLOCK_SIZE+size.CONTOUR_SIZE, size.BLOCK_SIZE-2*size.CONTOUR_SIZE, size.BLOCK_SIZE-2*size.CONTOUR_SIZE), 0)
        elif self.speed < 1 :
            pygame.draw.rect(screen, colors.RED, (self.pos[0]*size.BLOCK_SIZE+size.CONTOUR_SIZE, self.pos[1]*size.BLOCK_SIZE+size.CONTOUR_SIZE, size.BLOCK_SIZE-2*size.CONTOUR_SIZE, size.BLOCK_SIZE-2*size.CONTOUR_SIZE), 0)
        
class Plateau:
    def __init__(self, n_cols, n_rows, velocity = [7,8,12]):
        self.cases = []
        for i in range(n_cols*n_rows):
            self.cases.append(Case(i, n_cols, n_rows))
        self.add_velocity_effect(velocity)

            
    def add_velocity_effect(self, tab_cases_concerned):
        for i in tab_cases_concerned:
            self.cases[i].add_velocity()
    
    def draw_plateau(self, screen):
        for case in self.cases:
            case.draw_case(screen)
        
   
import pygame

# Initialiser Pygame
pygame.init()

# Définir la taille de l'écran
SCREEN_SIZE = (800, 600)

# Initialiser la fenêtre
screen = pygame.display.set_mode(SCREEN_SIZE)

plate = Plateau(5, 5)

plate.draw_plateau(screen)
# Mettre à jour l'affichage
pygame.display.update()

# Boucle d'événements
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quitter Pygame
pygame.quit()