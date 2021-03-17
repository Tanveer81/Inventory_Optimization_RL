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
import os
from rlib.shared.utils import Logger
import gym_example


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--experiment_name', default='dqn_demo', type=str)
    parser.add_argument('--num_episodes', default=5000, type=int)
    parser.add_argument('--seed', default=5214, type=int)
    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    # Training
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--eps_start', default=0.9, type=float)
    parser.add_argument('--eps_end', default=0.05, type=float)
    parser.add_argument('--eps_decay', default=200, type=int)
    parser.add_argument('--target_update', default=1000, type=int)
    parser.add_argument('--learning_rate', default=2.5e-4, type=float)
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
    parser.add_argument('--learn_every', default=4, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--buffer_size', default=int(1e7), type=int)
    parser.add_argument('--tau', default=1e-3, type=float)

    return parser


class PolicyNetwork(nn.Module):
    def __init__(self, n_materials=1, static_feature_length=5, n_actions=2,
                 hidden_dim_lstm=128, hidden_layers_mlp=[64, 32, 16], demand_embedding=3,
                 device='cpu'):
        super(PolicyNetwork, self).__init__()
        self.n_materials = n_materials
        self.static_feature_length = static_feature_length
        self.device = device  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm = LSTM(n_materials, hidden_dim_lstm, demand_embedding, 2, True).to(self.device)
        self.mlp = MLP(demand_embedding + static_feature_length, n_actions, hidden_layers_mlp,
                       activation="leakyRelu", batch_norm=False).to(self.device)

    def forward(self, state):
        static_state, seq_state = state[:, :, :self.static_feature_length], state[:, :, self.static_feature_length:]
        encoder_output = self.lstm(seq_state.reshape(seq_state.shape[0], -1, self.n_materials))
        hidden_features = torch.cat((encoder_output, static_state[:, 0, :]), dim=1)
        x = self.mlp(hidden_features)
        return x


class CustomGymEnvironment(GymEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start_env(self) -> None:
        """Override Helper to start an environment."""
        # self.env = gym.make(self._env_name, **kwargs)
        self.env.seed(self.seed)


class CustomDQNAgent(DQNAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save_state_dicts(self):
        """Override Save state dicts to file."""
        if not self.model_output_dir:
            return

        for sd in self.state_dicts:
            torch.save(
                sd[0].state_dict(),
                os.path.join(self.model_output_dir, "{}.pth".format(sd[1]))
            )

    def load_state_dicts(self):
        """Override Load state dicts from file."""
        if not self.model_output_dir:
            raise Exception("You must provide an input directory to load state dict.")

        for sd in self.state_dicts:
            sd[0].load_state_dict(
                torch.load(os.path.join(self.model_output_dir, "{}.pth".format(sd[1])))
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)

    output_dir = args.output_dir + '/' + args.experiment_name
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    random.seed(args.seed)

    mat_info = pd.read_csv("Data/Material_Information_q115.csv", sep=";", index_col="Material")
    hist_data = pd.read_csv("Data/Preprocessing/train_q115.csv")

    logger = Logger(path=output_dir, comment=None, verbosity='DEBUG',
                    experiment_name=args.experiment_name)

    config = {'hist_data': hist_data,
              'mat_info': mat_info,
              'random_reset': False,
              'sinusoidal_demand': args.sinusoidal_demand,
              'demand_satisfaction': args.demand_satisfaction,
              'past_demand': args.past_demand,
              'sine_type': args.sine_type,
              'noisy_demand': args.noisy_demand,
              'logger': logger
              }

    env = gym.make("stockManager-v0", **config)

    test_data = pd.read_csv("Data/Preprocessing/test_q115.csv")
    test_config = {'hist_data': test_data,
                   'mat_info': mat_info,
                   'random_reset': False,
                   'sinusoidal_demand': args.sinusoidal_demand,
                   'demand_satisfaction': args.demand_satisfaction,
                   'past_demand': args.past_demand,
                   'sine_type': args.sine_type,
                   'noisy_demand': args.noisy_demand,
                   'test': True,
                   'logger': logger
                   }

    test_env = gym.make("stockManager-v0", **test_config)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')
    # Get number of actions from gym action space
    # n_actions = env.action_space.n

    n_materials = 1

    policy_net = PolicyNetwork(n_materials=n_materials,
                               static_feature_length=3,
                               n_actions=2,
                               hidden_dim_lstm=args.hidden_dim_lstm,
                               hidden_layers_mlp=args.hidden_layers,
                               demand_embedding=args.demand_embedding,
                               device=device)

    target_net = PolicyNetwork(n_materials=n_materials,
                               static_feature_length=3,
                               n_actions=2,
                               hidden_dim_lstm=args.hidden_dim_lstm,
                               hidden_layers_mlp=args.hidden_layers,
                               demand_embedding=args.demand_embedding,
                               device=device)

    target_net.load_state_dict(policy_net.state_dict())

    new_hyperparameters = {
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "tau": args.tau,
        "learn_every": args.learn_every,
        "hard_update_every": args.target_update
    }

    dqn = CustomDQNAgent(
        state_size=args.past_demand + 3,
        action_size=2,
        qnetwork_local=policy_net,
        qnetwork_target=target_net,
        optimizer=None,
        new_hyperparameters=new_hyperparameters,
        seed=args.seed,
        device=device,
        model_output_dir=output_dir,
        opt_soft_update=False,
        opt_ddqn=False)

    e = CustomGymEnvironment(env=env,
                             algorithm=dqn,
                             seed=args.seed,
                             logger=logger,
                             gifs_recorder=None)
    e.train(num_episodes=args.num_episodes, max_t=None, add_noise=True,
            scores_window_size=100, save_every=1)

    print('\ntest\n')
    e_test = CustomGymEnvironment(env=test_env,
                                  algorithm=dqn,
                                  seed=args.seed,
                                  logger=logger,
                                  gifs_recorder=None)
    e_test.test(num_episodes=1, load_state_dicts=True, render=True)
