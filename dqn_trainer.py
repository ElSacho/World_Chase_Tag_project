#!/usr/bin/env python3
from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import random
import os

import torch
import torch.nn as nn
import torch.optim as optim


import collections
import numpy as np
from catState import CatState
from mouseState import MouseState
from gameEnv import GameEnv

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "ChaseTag"
MEAN_REWARD_BOUND = 100000

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000
HIDDEN_SIZE = 128

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.03

PARTICULAR_NAME ='nothing_5x5'
VISION = 4
N_ROWS = 5
N_COLS = 5
METHOD_TO_SPEND_TIME = None
CASES_TO_SPEND_TIME = None
METHOD_FOR_HOUSE = None
CASE_HOUSE = None
METHOD_FOR_WALL = None
CASE_WALL = None


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class CatAgent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset_cat()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon:
            action = random.randint(0, self.env.cat_action_space-1)
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            state_v = state_v.float()
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.cat_step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

class MouseAgent(MouseState):
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset_mouse()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon:
            action = random.randint(0, self.env.mouse_action_space-1)
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            state_v = state_v.float()
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        c_mouse[action] += 1
        # do step in the environment
        new_state, reward, is_done, _ = self.env.mouse_step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward



def calc_loss(batch, net, tgt_net, device="??"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    states_v = states_v.float()
    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_states_v = next_states_v.float()
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    i = 1
    while i <= 999:
        folder_name = 'models/models' + PARTICULAR_NAME + '_{:03d}'.format(i)
        if not os.path.exists(folder_name):
            break
        i += 1
        
    if i <= 998:
        next_folder_name = 'models/models' + PARTICULAR_NAME + '_{:03d}'.format(i)
        os.mkdir(next_folder_name)
        mouse_folder_name = os.path.join(next_folder_name, "mouse")
        cat_folder_name = os.path.join(next_folder_name, "cat")
        os.mkdir(mouse_folder_name)
        os.mkdir(cat_folder_name)
    else :
        raise('To many folders')
        
    # Le nom du fichier de sauvegarde
    filename = 'variables.txt'

    # Le dossier dans lequel le fichier sera cr????
    directory = 'models/models' + PARTICULAR_NAME + '_{:03d}'.format(i)

    # Chemin complet vers le fichier de sauvegarde
    filepath = os.path.join(directory, filename)

    # ??crire les variables dans le fichier de sauvegarde
    with open(filepath, 'w') as f:
        f.write('VISION={}\n'.format(VISION))
        f.write('N_ROWS={}\n'.format(N_ROWS))
        f.write('N_COLS={}\n'.format(N_COLS))
        if METHOD_TO_SPEND_TIME != None:
            f.write('METHOD_TO_SPEND_TIME="{}"\n'.format(METHOD_TO_SPEND_TIME))    
        f.write('CASES_TO_SPEND_TIME={}\n'.format(CASES_TO_SPEND_TIME))
        if METHOD_FOR_HOUSE != None:
            f.write('METHOD_FOR_HOUSE="{}"\n'.format(METHOD_FOR_HOUSE))
        f.write('CASE_HOUSE={}\n'.format(CASE_HOUSE))
        if METHOD_FOR_WALL != None:
            f.write('METHOD_FOR_WALL="{}"\n'.format(METHOD_FOR_WALL))
        f.write('CASE_WALL={}\n'.format(CASE_WALL))
    
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    parser.add_argument("-m", "--model_cat", required=False, default="old/PongNoFrameskip-v4-best_91.dat",
                        help="Model file to load")
    # parser.add_argument("-m", "--model1", required=False, default="PongNoFrameskip-v4-best_12.dat",
    #                     help="Model file to load")
    # parser.add_argument("-m2", "--model2", required=False, default="PongNoFrameskip-v4-best_11.dat",
    #                     help="Model file to load")
    args = parser.parse_args()
    
    writer_cat = SummaryWriter(comment="-cat" + args.env)
    writer_mouse = SummaryWriter(comment="-mouse" + args.env)
    
    device = torch.device("cuda" if args.cuda else "cpu")

    env = GameEnv(N_COLS, N_ROWS, vision = VISION, method = "speed", method_to_spend_time = METHOD_TO_SPEND_TIME, cases_to_spend_time = CASES_TO_SPEND_TIME, method_for_house = METHOD_FOR_HOUSE, case_house = CASE_HOUSE, method_for_wall = METHOD_FOR_WALL, case_wall = CASE_WALL)

    mouse_net = dqn_model.DQN(env.mouse_observation_space, HIDDEN_SIZE,
                        env.mouse_action_space).to(device)
    
    mouse_tgt_net = dqn_model.DQN(env.mouse_observation_space, HIDDEN_SIZE,
                        env.mouse_action_space).to(device)
    
    mouse_buffer = ExperienceBuffer(REPLAY_SIZE)
    mouse_agent = MouseAgent(env, mouse_buffer)
    
    cat_buffer = ExperienceBuffer(REPLAY_SIZE)
    cat_agent = CatAgent(env, cat_buffer)
    
    cat_net = dqn_model.DQN(env.cat_observation_space, HIDDEN_SIZE,
                        env.cat_action_space).to(device)

    
    cat_tgt_net = dqn_model.DQN(env.cat_observation_space, HIDDEN_SIZE,
                        env.cat_action_space).to(device)
    
    # cat_state = torch.load(args.model_cat, map_location=lambda stg, _: stg)
    
    # cat_net.load_state_dict(cat_state)

    
    epsilon = EPSILON_START

    # mouse initilisations
    mouse_optimizer = optim.Adam(mouse_net.parameters(), lr=LEARNING_RATE)
    mouse_total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    mouse_best_m_reward = None
    
    # cat initilisations
    cat_optimizer = optim.Adam(cat_net.parameters(), lr=LEARNING_RATE)
    cat_total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    cat_best_m_reward = None
    
    cat_state = env.reset_cat()
    
    c_mouse = collections.Counter()

    while True:
        frame_idx += 1
        # print(frame_idx)
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)
        

        #train the cat
        cat_reward = cat_agent.play_step(cat_net, epsilon, device=device)
        if cat_reward is not None:
            cat_total_rewards.append(cat_reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            cat_m_reward = np.mean(cat_total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
                    "eps %.2f, speed %.2f f/s" % (
                frame_idx, len(cat_total_rewards), cat_m_reward, epsilon,
                speed
            ))
            writer_cat.add_scalar("epsilon cat", epsilon, frame_idx)
            writer_cat.add_scalar("speed cat", speed, frame_idx)
            writer_cat.add_scalar("reward_100 cat", cat_m_reward, frame_idx)
            writer_cat.add_scalar("reward cat", cat_reward, frame_idx)
            if cat_best_m_reward is None or cat_best_m_reward < cat_m_reward:
                torch.save(cat_net.state_dict(), cat_folder_name +"/"+ args.env +
                            "-best_%.0f.dat" % cat_m_reward)
                if cat_best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        cat_best_m_reward, cat_m_reward))
                cat_best_m_reward = cat_m_reward
            if cat_m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(cat_buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            cat_tgt_net.load_state_dict(cat_net.state_dict())

        cat_optimizer.zero_grad()
        cat_batch = cat_buffer.sample(BATCH_SIZE)
        cat_loss_t = calc_loss(cat_batch, cat_net, cat_tgt_net, device=device)
        cat_loss_t.backward()
        cat_optimizer.step()

        #train the mouse
        mouse_reward = mouse_agent.play_step(mouse_net, epsilon, device=device)

        if mouse_reward is not None:
            mouse_total_rewards.append(mouse_reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mouse_m_reward = np.mean(mouse_total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
                "eps %.2f, speed %.2f f/s" % (
                frame_idx, len(mouse_total_rewards), mouse_m_reward, epsilon,
                speed
            ))
            writer_mouse.add_scalar("epsilon mouse", epsilon, frame_idx)
            writer_mouse.add_scalar("speed mouse", speed, frame_idx)
            writer_mouse.add_scalar("reward_100 cat", mouse_m_reward, frame_idx)
            writer_mouse.add_scalar("reward mouse", mouse_reward, frame_idx)
            c_mouse = collections.Counter()
            if mouse_best_m_reward is None or mouse_best_m_reward < mouse_m_reward:
                torch.save(mouse_net.state_dict(), mouse_folder_name +"/"+ args.env +
                        "-best_%.0f.dat" % mouse_m_reward)
                if mouse_best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        mouse_best_m_reward, mouse_m_reward))
                mouse_best_m_reward = mouse_m_reward
            if mouse_m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(mouse_buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            mouse_tgt_net.load_state_dict(mouse_net.state_dict())

        mouse_optimizer.zero_grad()
        mouse_batch = mouse_buffer.sample(BATCH_SIZE)
        mouse_loss_t = calc_loss(mouse_batch, mouse_net, mouse_tgt_net, device=device)
        mouse_loss_t.backward()
        mouse_optimizer.step()
    writer_cat.close()
    writer_mouse.close()
        