# Stability Of Deep Q Learning

- [Introduction](#Introduction)
- [Exécuter le code](#Compilation)


<a name="Introduction"></a>

## Introduction

Ce projet est un projet de Reinforcement Learning, qui comprend la création d'un environnement dans lequel deux agents peuvent s'affronter. L'objectif de l'agent "Cat" est d'attraper l'agent "Mouse", qui essaie donc de lui échapper. Le plateau de jeu peut être modifiable. Les cases rouges ralentissent les deux agents, les noirs sont des murs, tandis que les vertes sont des murs pour le le "Cat" mais une case normale pour le "Mouse". Les cases blanches sont normales. Les agents peuvent se déplacer selon 4 directions, (lorsque c'est possible) - à droite, à gauche, en haut et en bas - Le projet utilise du Deep Q Learning pour apprendre à nos agents à réaliser leurs tâches. 


Les enjeux de ce projet sont multiples 
* **Créer un environnement dans lequel les agents vont pouvoir s'affronter**. 
* **Réussir à entrainer nos agents**. 
* **Easy to read and understand source code**. 
* **Modifier la disposition des cases du plateau pour observer les réactions de nos agents, et ainsi étudier la stabilité des méthodes de Deep Q Learning**.

Nous avons répondu à ces enjeux, et les résultats obtenus sont disponibles dans le PDF présent sur le GitHub



<a name="Compilation"></a>

## Exécuter le code

* **Installer les librairies**

Il faut tout d'abord installer les librairies. Celles-ci sont : PyGame, tensorboardX, Torch, Numpy, collections, argparse, time.

Nous avons ensuite optimisé l'exécution du code. 

* **Pour entrainer un modèle**
Utiliser le fichier "dqn_trainer.py". Vous pouvez ensuite modifier les paramètres que vous souhaitez. En particulier :

- PARTICULAR_NAME: permet d'enregistrer votre modèle avec un nom particulier. Par exemple "model_no_walls". Par défault, l'algorithme créera un dossier numéroté dans le fichier "models/" avec comme numéro la version de ce modèle que vous êtes entrain de créer
- VISION: permet de définir le nombre de cases que les agents voients. Ce sera un carré de taille 2*VISION+1 centré sur l'agent.
- N_ROWS and N_COLS : définir la taille du plateau de jeu.
- METHOD_TO_SPEND_TIME : la méthode utilisée pour placer les cases qui ralentissent les Agents. Si la valeur est: "random" les cases seront générées aléatoirement. Sinon, cela dépendra de la valeur de CASES_TO_SPEND_TIME
- CASES_TO_SPEND_TIME : Si cette variable prend la valeur "None", aucune case ne sera un ralentisseur. Sinon, les cases indiquées comme devant être des ralentisseurs le seront. 
- METHOD_FOR_HOUSE : idem pour ajouter une case "Maison" pour la "Mouse"
- CASE_HOUSE : dem pour ajouter une case "Maison" pour la "Mouse"
- METHOD_FOR_WALL : idem pour ajouter une case "Wall"
- CASE_WALL : idem pour ajouter une case "Wall"

* **Afficher les résultats d'un modèle**

Cela se passe dans le fichier "dqn_test.py". Il suffit d'indiquer la valeur de PARTICULAR_NAME souhaitée pour automatiquement load le dernier modèle crée avec ce PARTICULAR_NAME. 

La varible DRAW permet d'afficher le plateau de jeu lorsque sa valeur est True, et ne display rien sinon.

La variable LOAD_MODEL permet de load le plateau de jeu d'entrainement sur lequel nos agents ont été entrainés lorsqu'elle est True. Lorsque cette variable est False, ce sont les paramètres rentrés par l'utilsateur aux cases suivantes qui définieront le plateau de jeu. 

