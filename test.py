import random
import os

filepath = 'models/models_just_vision_003/variables.txt'

with open(filepath, 'r') as f:
    for line in f:
        exec(line.strip())

# Afficher les variables lues du fichier de sauvegarde
print(PARTICULAR_NAME)
print(type(VISION))
print(type(None))


