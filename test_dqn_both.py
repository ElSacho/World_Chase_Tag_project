#!/usr/bin/env python3
import time
import argparse
import numpy as np

import torch

from lib import dqn_model

import collections
import numpy as np
from gameEnv import GameEnv

DEFAULT_ENV_NAME = "Env Chase Tag"
FPS = 25
HIDDEN_SIZE = 128
DRAW = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-mM", "--modelMouse", required=False, default="old/Cahse_tag-best_16.dat",
    #                     help="Model file to load")
    parser.add_argument("-mM", "--modelMouse", required=False, default="old/Cahse_tag-best_4.dat",
                        help="Model file to load")
    parser.add_argument("-mC", "--modelCat", required=False, default="old/PongNoFrameskip-v4-best_91.dat",
                        help="Model file to load")
    # parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
    #                     help="Environment name to use, default=" +
    #                          DEFAULT_ENV_NAME)
    # parser.add_argument("-r", "--record", help="Directory for video")
    # parser.add_argument("--no-vis", default=True, dest='vis',
    #                     help="Disable visualization",
    #                     action='store_false')
    
    args = parser.parse_args()

    env = GameEnv(5,5, vision = 1, method = "human")
    
    cat_net = dqn_model.DQN(env.cat_observation_space, HIDDEN_SIZE,
                        env.cat_action_space)
    
    cat_state = torch.load(args.modelCat, map_location=lambda stg, _: stg)
    
    cat_net.load_state_dict(cat_state)
    
    
    mouse_net = dqn_model.DQN(env.mouse_observation_space, HIDDEN_SIZE,
                        env.mouse_action_space)
    
    mouse_state = torch.load(args.modelMouse, map_location=lambda stg, _: stg)
    
    mouse_net.load_state_dict(mouse_state)
    
    while True :
    
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
                break
            state_v = torch.tensor(np.array([mouse_state], copy=False))
            state_v = state_v.float()
            q_vals = mouse_net(state_v).data.numpy()[0]
            action = np.argmax(q_vals)
            c_mouse[action] += 1
            mouse_state, reward, done, _ = env.mouse_step(action)
            # print(reward)
            total_reward_mouse += reward
            if done:
                break
            # if args.vis:
            #     delta = 1/FPS - (time.time() - start_ts)
            #     if delta > 0:
            #         time.sleep(delta)
        print("Total reward cat: %.2f" % total_reward_cat)
        print("Action counts cat:", c_cat)
        print("Total reward mouse : %.2f" % total_reward_mouse)
        print("Action counts mouse:", c_mouse)
        # if args.record:
        #     env.env.close()

