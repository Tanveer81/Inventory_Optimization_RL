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
    parser = argparse.ArgumentParser('Set environment and train-test configuration', add_help=False)

    parser.add_argument('--experiment_name', default='dqn_demo', type=str, help="")
    parser.add_argument('--num_episodes', default=5000, type=int, help="Number of episodes the dqn agent will be trained.")
    parser.add_argument('--seed', default=5214, type=int, help="random seed for all libraries involved to recreate experiments. Kept at default.")
    parser.add_argument('--output_dir', default='output', help='path where to save model weights, empty for no saving')
    parser.add_argument('--env', default='stockManager-v1', type=str, choices=('stockManager-v0', 'stockManager-v1'), help="stockManager-v1 is the best working environment")
    # Training
    parser.add_argument('--gamma', default=0.99, type=float, help="discount factor for dqn algorithm")
    parser.add_argument('--target_update', default=1000, type=int, help="target network update frequency")
    parser.add_argument('--learning_rate', default=2.5e-4, type=float, help="lr for optimizing the agent")
    parser.add_argument('--train', default=False, type=bool, help="run training")
    parser.add_argument('--test', default=False, type=bool, help="run testing")
    parser.add_argument('--resume', default=False, type=bool, help="resume training from last epoch")
    parser.add_argument('--past_demand', default=3, type=int, help="number of days past demand and stock level to use for agent's training")
    parser.add_argument("--hidden_layers", nargs="*", type=int, default=[256, 128], help="hidden layers for mlp head of the policy and target network")
    parser.add_argument('--demand_embedding', default=3, type=int, help="embedding dimension for the history features from sequence model(LSTM) inside policy network")
    parser.add_argument('--hidden_dim_lstm', default=128, type=int, help="dimension of the hidden embedding of sequence model(LSTM) inside policy network")
    parser.add_argument('--learn_every', default=4, type=int, help="policy network update frequency")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size of the replay memory of dqn")
    parser.add_argument('--buffer_size', default=int(1e7), type=int, help="ReplayBuffer size of dqn")
    parser.add_argument('--tau', default=1e-3, type=float, help="Interpolation parameter/ weighted update rate of target network")
    parser.add_argument('--opt_soft_update', default=False, type=bool, help="")
    parser.add_argument('--opt_ddqn', default=False, type=bool, help="")
    parser.add_argument("--cuda_visible_device", nargs="*", type=int, default=None,help="list of cuda visible devices")
    parser.add_argument('--inventory_weight', default=1, type=int, help="scale default inventory cost")
    parser.add_argument('--stock_out_weight', default=1, type=int, help="scale default stock out cost")
    parser.add_argument('--hack_test', default=False, type=bool, help="wait order arrival time before taking next option for testing")
    parser.add_argument('--hack_train', default=False, type=bool, help="wait order arrival time before taking next option for training")
    parser.add_argument('--evaluate_train', default=False, type=bool, help="evaluate agent on training data")
    parser.add_argument('--material_name', default='Q115', type=str,
                        choices= ['B120BP', 'B120', 'Q120', 'TA2J6500', 'Q115', 'Q2100H', 'Q3015'], help="which material to train/ test the agent on")
    parser.add_argument('--immediate_action_train', default=False, type=bool, help="Add reorder amount immediately without delay only for training")
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
    parser = argparse.ArgumentParser('Inventory Optimization Training and Testing Script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    output_dir = output_dir_logger = args.output_dir + '/' + args.experiment_name
    if args.test and not args.train:
        output_dir_logger = args.output_dir+'_test' + '/' + args.experiment_name
        name = '_hack_test' if args.hack_test else '_test'
        if args.evaluate_train:
            name = name + '_evaluate_train'
        output_dir_logger = output_dir_logger+name
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    random.seed(args.seed)

    mat_info = pd.read_csv("Data/Material_Information.csv", sep=";", index_col="Material")
    mat_info = mat_info.loc[[args.material_name]]
    hist_data = pd.read_csv("Data/Preprocessing/train.csv")
    hist_data = hist_data[[args.material_name]]

    logger = Logger(path=output_dir_logger, comment=None, verbosity='DEBUG',
                    experiment_name=args.experiment_name)

    config = {'exp_name': args.experiment_name + '_train' if args.evaluate_train else '_test',
              'hist_data': hist_data,
              'mat_info': mat_info,
              'random_reset': False,
              'past_demand': args.past_demand,
              'logger': logger,
              'inventory_weight': args.inventory_weight,
              'stock_out_weight': args.stock_out_weight,
              'hack_train': args.hack_train,
              'immediate_action_train': args.immediate_action_train
              }
    if args.env == 'stockManager-v0':
        config.pop('inventory_weight', None)
        config.pop('stock_out_weight', None)
        config.pop('hack_train', None)
        config.pop('exp_name', None)
    env = gym.make(args.env, **config)

    test_data = pd.read_csv("Data/Preprocessing/test.csv")
    test_data = test_data[[args.material_name]]
    test_config = {'exp_name': args.experiment_name + ('_train' if args.evaluate_train else '_test'),
                   'hist_data': hist_data if args.evaluate_train else test_data,
                   'mat_info': mat_info,
                   'random_reset': False,
                   'past_demand': args.past_demand,
                   'test': True,
                   'logger': logger,
                   'hack_test': args.hack_test,
                   }
    if args.env == 'stockManager-v0':
        test_config.pop('hack_test', None)
        test_config.pop('exp_name', None)
    test_env = gym.make(args.env, **test_config)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')


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
        "hard_update_every": args.target_update,
        'opt_soft_update': args.opt_soft_update,
        'opt_ddqn': args.opt_ddqn
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

    if args.resume:
        dqn.load_state_dicts()

    if args.train:
        e = CustomGymEnvironment(env=env,
                                 algorithm=dqn,
                                 seed=args.seed,
                                 logger=logger,
                                 gifs_recorder=None)
        e.train(num_episodes=args.num_episodes, max_t=None, add_noise=True,
                scores_window_size=100, save_every=1)

    if args.test:
        print('\ntest\n')
        e_test = CustomGymEnvironment(env=test_env,
                                      algorithm=dqn,
                                      seed=args.seed,
                                      logger=logger,
                                      gifs_recorder=None)
        e_test.test(num_episodes=1, load_state_dicts=True, render=True)

    logger.close()