#!/usr/bin/env python3
import time
import argparse
import numpy as np

import torch

from lib import dqn_model

import collections
import numpy as np
from catState import CatState
from gameEnv import GameEnv

DEFAULT_ENV_NAME = "Env Chase Tag"
FPS = 25
HIDDEN_SIZE = 128
DRAW = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=False, default="PongNoFrameskip-v4-best_8.dat",
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
    
    net = dqn_model.DQN(env.cat_observation_space, HIDDEN_SIZE,
                        env.cat_action_space)
    
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    
    net.load_state_dict(state)
    
    while True :
    
        state = env.reset_cat()
        total_reward = 0.0
        c = collections.Counter()
        while True:
            start_ts = time.time()
            if DRAW:
                env.draw()
            state_v = torch.tensor(np.array([state], copy=False))
            state_v = state_v.float()
            q_vals = net(state_v).data.numpy()[0]
            action = np.argmax(q_vals)
            c[action] += 1
            state, reward, done, _ = env.cat_step(action)
            total_reward += reward
            if done:
                break
            # if args.vis:
            #     delta = 1/FPS - (time.time() - start_ts)
            #     if delta > 0:
            #         time.sleep(delta)
        print("Total reward: %.2f" % total_reward)
        print("Action counts:", c)
        # if args.record:
        #     env.env.close()

