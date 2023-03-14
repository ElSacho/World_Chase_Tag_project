#!/usr/bin/env python3
import time
import argparse
import numpy as np

import os
import re

import torch

from lib import dqn_model

import collections
import numpy as np
from gameEnv import GameEnv

from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = "Env Chase Tag"

FPS = 25
HIDDEN_SIZE = 128
DRAW = True

LOAD_MODEL = True
PARTICULAR_NAME ='_all_features_6x6'
PARTICULAR_NAME = "nothing_5x5"
PARTICULAR_NAME ='all_features_7x7'
PARTICULAR_NAME ='wall_in_the_center'
PARTICULAR_NAME ='3 layers instead of 2 layers'
PARTICULAR_NAME ='_5x5_walls_and_red'
PARTICULAR_NAME ='_5x5_walls_and_red'
VISION = 4
N_ROWS = 7
N_COLS = 7
METHOD_TO_SPEND_TIME = None
CASES_TO_SPEND_TIME = None
METHOD_FOR_HOUSE = None
CASE_HOUSE = None
METHOD_FOR_WALL = None
CASE_WALL = None


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # on ouvre automatiquement le dernier model crée
    directory = "models"
    
    if PARTICULAR_NAME == '':
        files = os.listdir(directory)
    else :
        files = [f for f in os.listdir(directory) if PARTICULAR_NAME in f]

    sorted_files = sorted(files)
    last_file = sorted_files[-1]

    new_directory = os.path.join(directory, last_file)
    
    mouse_directory = directory+'/'+last_file +"/mouse"
    files = os.listdir(mouse_directory)
    mouse_name = max(files, key=lambda x: int(re.search(r'\d+', x).group()))
    mouse_name = os.path.join(mouse_directory, mouse_name)

    cat_directory = directory+'/'+last_file +"/cat"
    files = os.listdir(cat_directory)
    cat_name = max(files, key=lambda x: int(re.search(r'\d+', x).group()))
    cat_name = os.path.join(cat_directory, cat_name)
    
    if LOAD_MODEL:
        filepath = os.path.join(new_directory, 'variables.txt')
        with open(filepath, 'r') as f:
            for line in f:
                exec(line.strip())
    

    parser.add_argument("-mM", "--modelMouse", required=False, default=mouse_name,
                        help="Model file to load")
    parser.add_argument("-mC", "--modelCat", required=False, default=cat_name,
                        help="Model file to load")

    
    args = parser.parse_args()
    
    writer_cat = SummaryWriter(comment="-test_cat" )
    writer_mouse = SummaryWriter(comment="-test_mouse" )

    env = GameEnv(N_COLS, N_ROWS, vision = VISION, method = "human", method_to_spend_time = METHOD_TO_SPEND_TIME, cases_to_spend_time = CASES_TO_SPEND_TIME, method_for_house = METHOD_FOR_HOUSE, case_house = CASE_HOUSE, method_for_wall = METHOD_FOR_WALL, case_wall = CASE_WALL)
    
    cat_net = dqn_model.DQN(env.cat_observation_space, HIDDEN_SIZE,
                        env.cat_action_space)
    
    cat_state = torch.load(args.modelCat, map_location=lambda stg, _: stg)
    
    cat_net.load_state_dict(cat_state)
    
    
    mouse_net = dqn_model.DQN(env.mouse_observation_space, HIDDEN_SIZE,
                        env.mouse_action_space)
    
    mouse_state = torch.load(args.modelMouse, map_location=lambda stg, _: stg)
    
    mouse_net.load_state_dict(mouse_state)
    
    frame_idx = 0
    
    tab_mouse_reward = []
    tab_cat_reward = []
    
    wins = 0
    games = 0
    
    while True :
        
        frame_idx += 1
    
        cat_state = env.reset_cat()
        mouse_state = env.get_state_mouse()
        
        total_reward_cat = 0.0
        c_cat = collections.Counter()
        
        total_reward_mouse = 0.0
        c_mouse = collections.Counter()
        while True:
            start_ts = time.time()
            if DRAW:
                env.draw()
            state_v = torch.tensor(np.array([cat_state], copy=False))
            state_v = state_v.float()
            q_vals = cat_net(state_v).data.numpy()[0]
            action = np.argmax(q_vals)
            c_cat[action] += 1
            cat_state, reward, done, _ = env.cat_step(action)
            total_reward_cat += reward
        
            if done:
                if total_reward_cat < 1:
                    wins += 1
                games +=1
                tab_mouse_reward.append(total_reward_mouse)
                tab_cat_reward.append(total_reward_cat)
                mouse_m_reward = np.mean(tab_mouse_reward[-100:])
                cat_m_reward = np.mean(tab_cat_reward[-100:])
                writer_cat.add_scalar("reward_100 cat", cat_m_reward, frame_idx)
                writer_mouse.add_scalar("reward_100 cat", mouse_m_reward, frame_idx)
                break
            
            if DRAW:
                env.draw()
            state_v = torch.tensor(np.array([mouse_state], copy=False))
            state_v = state_v.float()
            q_vals = mouse_net(state_v).data.numpy()[0]
            action = np.argmax(q_vals)
            c_mouse[action] += 1
            mouse_state, reward, done, _ = env.mouse_step(action)
            # print(reward)
            total_reward_mouse += reward

            if done:
                if total_reward_cat < 1:
                    wins += 1
                games +=1
                tab_mouse_reward.append(total_reward_mouse)
                tab_cat_reward.append(total_reward_cat)
                mouse_m_reward = np.mean(tab_mouse_reward[-100:])
                cat_m_reward = np.mean(tab_cat_reward[-100:])
                writer_cat.add_scalar("reward_100 cat", cat_m_reward, frame_idx)
                writer_mouse.add_scalar("reward_100 cat", mouse_m_reward, frame_idx)
                break
            # if args.vis:
            #     delta = 1/FPS - (time.time() - start_ts)
            #     if delta > 0:
            #         time.sleep(delta)
        print("Total reward cat: %.2f" % total_reward_cat)
        print("Action counts cat:", c_cat)
        print("Total reward mouse : %.2f" % total_reward_mouse)
        print("Action counts mouse:", c_mouse)
        pourc_wins = float(wins/games)
        print("‰ of mouse wins : %.2f" % pourc_wins)
        # if args.record:
        #     env.env.close()
    
    writer_cat.close()
    writer_mouse.close()
