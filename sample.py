#!/usr/bin/env python
# encoding: utf-8

import gym
import gym_example
import argparse
import pandas as pd
import numpy as np
import random

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--experiment_name', default='dqn_demo', type=str)
    parser.add_argument('--num_episodes', default=5000, type=int)
    parser.add_argument('--seed', default=5214, type=int)
    parser.add_argument('--output_dir', default='../output',
                        help='path where to save, empty for no saving')
    # Training
    parser.add_argument('--gamma', default=0.999, type=float)
    parser.add_argument('--eps_start', default=0.9, type=float)
    parser.add_argument('--eps_end', default=0.05, type=float)
    parser.add_argument('--eps_decay', default=200, type=int)
    parser.add_argument('--target_update', default=10, type=int)
    parser.add_argument('--sgd_regressor_lr', default=1e-5, type=float)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--epsilon', default=0.1, type=float)
    parser.add_argument('--epsilon_decay', default=0.99, type=float)
    parser.add_argument('--train', default=False, type=bool)
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--sinusoidal_demand', default=False, type=bool)
    parser.add_argument('--sine_type', default=3, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--replay_batch_size', default=128, type=int)
    parser.add_argument('--demand_satisfaction', default=False, type=bool)
    parser.add_argument('--past_demand', default=3, type=int)
    parser.add_argument('--noisy_demand', default=False, type=bool)
    parser.add_argument("--hidden_layers", nargs="*", type=int, default=[256, 128])
    parser.add_argument('--demand_embedding', default=3, type=int)
    parser.add_argument('--hidden_dim_lstm', default=128, type=int)

    return parser

def run_one_episode (env, verbose=False):
    env.reset()
    sum_reward = 0
    env.MAX_STEPS = 1750
    for i in range(env.MAX_STEPS):
        action = [env.action_space.sample()]

        if verbose:
            print("action:", action)

        state, reward, done = env.step(action, i)
        sum_reward += reward[0]

        # if verbose:
        #     env.render()

        if done:
            if verbose:
                print("done @ step {}".format(i))

            break

    if verbose:
        print("cumulative reward", sum_reward)

    return sum_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    np.random.seed(args.seed)
    random.seed(args.seed)

    BATCH_SIZE = args.replay_batch_size
    GAMMA = args.gamma
    EPS_START = args.eps_start
    EPS_END = args.eps_end
    EPS_DECAY = args.eps_decay
    TARGET_UPDATE = args.target_update
    num_episodes = args.num_episodes
    hidden_layers = args.hidden_layers
    n_materials = 1

    mat_info = pd.read_csv("Data/Material_Information_q115.csv", sep=";", index_col="Material")
    hist_data = pd.read_csv("Data/Preprocessing/train_q115.csv")

    config = {'hist_data': hist_data, 'mat_info': mat_info, 'random_reset': False,
              'sinusoidal_demand': args.sinusoidal_demand,
              'demand_satisfaction': args.demand_satisfaction,
              'past_demand': args.past_demand,
              'sine_type': args.sine_type,
              'noisy_demand': args.noisy_demand
              }

    env = gym.make("stockManager-v0", **config)
    sum_reward = run_one_episode(env, verbose=True)

    # next, calculate a baseline of rewards based on random actions
    # (no policy)
    history = []

    for _ in range(10000):
        sum_reward = run_one_episode(env, verbose=False)
        history.append(sum_reward)

    avg_sum_reward = sum(history) / len(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))
