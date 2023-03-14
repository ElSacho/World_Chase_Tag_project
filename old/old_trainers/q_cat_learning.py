#!/usr/bin/env python3
import collections
import numpy as np
from catState import CatState
from gameEnv import GameEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random

GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = GameEnv(5,5, vision = 0, method = "speed")
        self.cat_state = self.env.reset_cat()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = random.randint(0, self.env.cat_action_space-1)
        old_state = self.cat_state
        new_state, reward, is_done, _ = self.env.cat_step(action)
        self.cat_state = self.env.reset_cat() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.cat_action_space):
            action_value = self.values[(tuple(state), action)]
            print(f'action value : {action_value} our laction {action}')
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
                print(f'action choisie : {action}')
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        old_v = self.values[(tuple(s), a)]
        self.values[(tuple(s), a)] = old_v * (1-ALPHA) + new_v * ALPHA

    def play_episode(self, env):
        total_reward = 0.0
        state = self.env.reset_cat()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.cat_step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = GameEnv(5,5, vision = 0, method = "speed")
    agent = Agent()
  #  writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        print(iter_no)
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
       # writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        if reward > 12:
            print("Solved in %d iterations!" % iter_no)
            break
   # writer.close()