#!/usr/bin/env python
# encoding: utf-8

from ray.tune.registry import register_env
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil
import gym
import gym_example
import argparse
import pandas as pd
import numpy as np
import random
from gym_example.envs.TD_stock_manager import StockManager

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

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('DETR training and evaluation script',
    #                                  parents=[get_args_parser()])
    # args = parser.parse_args()
    # print(args)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    #
    # BATCH_SIZE = args.replay_batch_size
    # GAMMA = args.gamma
    # EPS_START = args.eps_start
    # EPS_END = args.eps_end
    # EPS_DECAY = args.eps_decay
    # TARGET_UPDATE = args.target_update
    # num_episodes = args.num_episodes
    # hidden_layers = args.hidden_layers
    # n_materials = 1
    #
    # mat_info = pd.read_csv("Data/Material_Information_q115.csv", sep=";", index_col="Material")
    # hist_data = pd.read_csv("Data/Preprocessing/train_q115.csv")



    # env = gym.make("stockManager-v0", **config)
    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    # select_env = "stockManager-v0"
    # register_env(select_env, lambda config: StockManager())

    select_env = StockManager()


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["train_batch_size"] = 1750

    # env_config = {'hist_data': hist_data,
    #           'mat_info': mat_info,
    #           'random_reset': False,
    #           'sinusoidal_demand': args.sinusoidal_demand,
    #           'demand_satisfaction': args.demand_satisfaction,
    #           'past_demand': args.past_demand,
    #           'sine_type': args.sine_type,
    #           'noisy_demand': args.noisy_demand
    #           }
    agent = ppo.PPOTrainer(config, env=StockManager)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 5

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))


    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())


    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = StockManager()

    state = env.reset()
    sum_reward = 0
    n_step = 2000

    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        # env.render()

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0
