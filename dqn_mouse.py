#!/usr/bin/env python3
# from lib import wrappers
from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import random

import torch
import torch.nn as nn
import torch.optim as optim


import collections
import numpy as np
from catState import CatState
from gameEnv import GameEnv



DEFAULT_ENV_NAME = "Cahse_tag"
MEAN_REWARD_BOUND = 100

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000
HIDDEN_SIZE = 128

EPSILON_DECAY_LAST_FRAME = 300000
EPSILON_START = 1.0
EPSILON_FINAL = 0.03


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

class MouseAgent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset_mouse()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        global c_mouse 
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



def calc_loss(batch, net, tgt_net, device="cpu"):
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
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    parser.add_argument("-m", "--model", required=False, default="old/PongNoFrameskip-v4-best_91.dat",
                        help="Model file to load")
    # parser.add_argument("-m", "--model1", required=False, default="PongNoFrameskip-v4-best_12.dat",
    #                     help="Model file to load")
    # parser.add_argument("-m2", "--model2", required=False, default="PongNoFrameskip-v4-best_11.dat",
    #                     help="Model file to load")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = GameEnv(5,5, vision = 1, method = "speed")

    net = dqn_model.DQN(env.cat_observation_space, HIDDEN_SIZE,
                        env.cat_action_space).to(device)
    
    # state = torch.load(args.model1, map_location=lambda stg, _: stg)
    
    # net.load_state_dict(state)
    
    tgt_net = dqn_model.DQN(env.cat_observation_space, HIDDEN_SIZE,
                        env.cat_action_space).to(device)
    
    # state2 = torch.load(args.model2, map_location=lambda stg, _: stg)
    
    # tgt_net.load_state_dict(state2)
    # writer = SummaryWriter(comment="-" + args.env)
    # print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = MouseAgent(env, buffer)
    cat_agent = CatAgent(env, buffer)
    
    cat_net = dqn_model.DQN(env.cat_observation_space, HIDDEN_SIZE,
                        env.cat_action_space)
    
    cat_state = torch.load(args.model, map_location=lambda stg, _: stg)
    
    cat_net.load_state_dict(cat_state)
    
    
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None
    
    c_mouse = collections.Counter()

    while True:
        frame_idx += 1
        # print(frame_idx)
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)
        
        cat_state = env.reset_cat()
        
        cat_state_v = torch.tensor(np.array([cat_state], copy=False))
        cat_state_v = cat_state_v.float()
        cat_q_vals = cat_net(cat_state_v).data.numpy()[0]
        cat_action = np.argmax(cat_q_vals)
        cat_state, _, _, _ = env.cat_step(cat_action)
        

        reward = agent.play_step(net, epsilon, device=device)

        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), m_reward, epsilon,
                speed
            ))
            print("Action counts mouse:", c_mouse)
            c_mouse = collections.Counter()
            # writer.add_scalar("epsilon", epsilon, frame_idx)
            # writer.add_scalar("speed", speed, frame_idx)
            # writer.add_scalar("reward_100", m_reward, frame_idx)
            # writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env +
                           "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    # writer.close()