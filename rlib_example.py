import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from network_utils import MLP, LSTM
import argparse
from rlib.algorithms.dqn import DQNAgent
from rlib.environments.gym import GymEnvironment
import gym
import gym_example

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


class PolicyNetwork(nn.Module):
    def __init__(self, n_materials=1, static_feature_length=5, n_actions=2,
                 hidden_dim_lstm=128, hidden_layers_mlp=[64, 32, 16], demand_embedding=3):
        super(PolicyNetwork, self).__init__()
        self.n_materials = n_materials
        self.static_feature_length = static_feature_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm = LSTM(n_materials, hidden_dim_lstm, demand_embedding, 2, True).to(self.device)
        self.mlp = MLP(demand_embedding + static_feature_length, n_actions, hidden_layers_mlp,
                       activation="leakyRelu", batch_norm=False).to(self.device)

    def forward(self, state):
        static_state, seq_state = state[:, :self.static_feature_length], state[:,
                                                                         self.static_feature_length:]
        encoder_output = self.lstm(seq_state.reshape(seq_state.shape[0], -1, self.n_materials))
        hidden_features = torch.cat((encoder_output, static_state), dim=1)
        X = self.mlp(hidden_features)
        return X

# def env_creator(env_name):
#     if env_name == 'stockMan-v0':
#         from src.TD_stock_manager import StockManager as env
#     else:
#         raise NotImplementedError
#     return env

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
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')
    # Get number of actions from gym action space
    # n_actions = env.action_space.n

    policy_net = PolicyNetwork(n_materials=n_materials,
                               static_feature_length=3,
                               n_actions=2, hidden_dim_lstm=args.hidden_dim_lstm,
                               hidden_layers_mlp=hidden_layers,
                               demand_embedding=args.demand_embedding)
    target_net = PolicyNetwork(n_materials=n_materials,
                               static_feature_length=3,
                               n_actions=2, hidden_dim_lstm=args.hidden_dim_lstm,
                               hidden_layers_mlp=hidden_layers,
                               demand_embedding=args.demand_embedding)

    target_net.load_state_dict(policy_net.state_dict())

    dqn = DQNAgent(
        args.past_demand + 3, 2,
        qnetwork_local=policy_net,
        qnetwork_target=target_net,
    )

    e = GymEnvironment(env, dqn)
    e.train()
    e.test()
