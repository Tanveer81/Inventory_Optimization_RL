import math
import random
from abc import ABC
import numpy as np
import gym
import pandas as pd


class StockManagerSingleAction(gym.Env, ABC):

    def __init__(self, hist_data=None, mat_info=None, random_reset=False, sinusoidal_demand=False,
                 demand_satisfaction=False, past_demand=3, sine_type=3, noisy_demand=False,
                 test=False, logger=None, stock_out_weight=1, inventory_weight=1,
                 hack_train=False, hack_test=False):

        super(StockManagerSingleAction, self).__init__()
        self.hack_train = hack_train
        self.hack_test = hack_test
        self.stock_out_weight = stock_out_weight
        self.inventory_weight = inventory_weight
        self.noisy_demand = noisy_demand
        self.sine_type = sine_type
        self.past_demand = past_demand
        self.demand_sat = demand_satisfaction
        self.sinusoidal_demand = sinusoidal_demand
        self.random_reset = random_reset
        if mat_info is None:
            mat_info = self.mat_info = pd.read_csv("Data/Material_Information_q115.csv", sep=";",
                                                   index_col="Material")
        else:
            self.mat_info = mat_info
        if hist_data is None:
            hist_data = self.history = pd.read_csv("Data/Preprocessing/train_q115.csv")
        else:
            self.history = hist_data

        self.reorder = mat_info.loc[hist_data.columns, 'Order_Volume'].values
        self.inv_costs = mat_info.loc[hist_data.columns, 'Inventory_Costs'].values
        self.stock_out_costs = mat_info.loc[hist_data.columns, 'Stock_Out_Costs'].values
        self.delivery_time = mat_info.loc[hist_data.columns, 'Delivery_Time_days'].values
        self.storage_cap = mat_info.loc[hist_data.columns, 'Order_Volume'].values * 4

        # initialize an empty Demand History (demand_history)
        self.monitor_timestep = [self.past_demand for _ in range(len(mat_info))]
        self.demand_satisfaction = [0 for _ in range(len(mat_info))]

        # initialize current Stock Level (stock_level) : can be randomized
        self.stock_level = mat_info.loc[hist_data.columns, 'Order_Volume'].values

        # define observation space (currently for only one material)
        '''
            stock level
            unfullfilled demand
            days to next reorder
            current demand
            previous days demand
            previous two days' demand
        '''
        list1 = [self.storage_cap[0], self.storage_cap[0]]
        list2 = [100000] * self.past_demand
        self.observation_space = gym.spaces.Box(low=np.array([0] * (self.past_demand + 2)),
                                                high=np.array(list1 + list2),
                                                dtype=np.float32)

        # define action space (currently for only one material)
        self.action_space = gym.spaces.Discrete(2)

        self.time_step = 0
        self.reward = 0
        self.action = 0

        self.test = test
        self.logger = logger

        # define observation space (currently for only one material)
        '''
            stock level
            unfullfilled demand
            days to next reorder
            current demand
            previous days demand
            previous two days' demand
        '''
        self.observation_space = gym.spaces.Box(
            low=np.array([[float('-inf')] * (self.past_demand + 3)]),
            high=np.array([[float('inf')] * (self.past_demand + 3)]),
            dtype=np.float32)

        # define action space (currently for only one material)
        self.action_space = gym.spaces.Discrete(2)

    def reward_for_one_inner_timestep(self, mat_i):

        # demand_sat = self.demand_satisfaction[mat_i] if self.demand_sat else 0
        reward = self.stock_out_weight * self.stock_out_costs[mat_i] * min(0, self.stock_level[mat_i]) \
                 - self.inventory_weight * self.inv_costs[mat_i] * max(0, self.stock_level[mat_i])

        return reward

    def update_stock_level_and_reward(self, actions):
        """
        defines the transition function and the reward function
        :return:
        """
        actions = [actions]  # TODO: Hack for compatibility, change if possible
        avg_rewards_all_materials = []
        # loop over all materials
        for i in range(len(self.stock_level)):

            if actions[i] == 1:
                self.stock_level[i] = self.stock_level[i] + self.reorder[i]

            # for the next n days equal to delivery time, update the stock level and reward.
            rewards = []
            monitor_time = self.monitor_timestep[i]

            if self.test and actions[i] == 0 and self.hack_test:
                # for each inner time step, introduce the current demand and calculate the reward.
                self.stock_level[i] = self.stock_level[i] - self.history.iloc[monitor_time, i]
                rewards.append(self.reward_for_one_inner_timestep(i))

                if self.logger:
                    self.logger.add_scalar(
                        'stock_level', self.stock_level[i], self.monitor_timestep[i]
                    )
                    self.logger.add_scalar(
                        'reward', self.reward_for_one_inner_timestep(i), self.monitor_timestep[i]
                    )
                    self.logger.add_scalar(
                        'effective_action', actions[i], self.monitor_timestep[i]
                        # TODO: remove hack
                    )
                self.monitor_timestep[i] += 1

            elif actions[i] == 0 and self.hack_train:
                temp_stock = self.stock_level[i] - self.history.iloc[monitor_time, i]
                for j in range(self.delivery_time[i]):
                    # for each inner time step, introduce the current demand and calculate the reward.
                    self.stock_level[i] = self.stock_level[i] - self.history.iloc[
                        j + monitor_time, i]
                    rewards.append(self.reward_for_one_inner_timestep(i))
                    self.monitor_timestep[i] += 1

                self.monitor_timestep[i] = self.monitor_timestep[i] - self.delivery_time[i] + 1
                self.stock_level[i] = temp_stock
                if self.logger and self.test:
                    self.logger.add_scalar(
                        'stock_level', self.stock_level[i], self.monitor_timestep[i]
                    )
                    self.logger.add_scalar(
                        'reward', avg_rewards_all_materials.append(np.mean(rewards)),
                        self.monitor_timestep[i]
                    )
                    self.logger.add_scalar(
                        'effective_action', 0, self.monitor_timestep[i]
                        # TODO: remove hack
                    )

            else:
                for j in range(self.delivery_time[i]):
                    # for each inner time step, introduce the current demand and calculate the reward.
                    self.stock_level[i] = self.stock_level[i] - self.history.iloc[
                        j + monitor_time, i]
                    rewards.append(self.reward_for_one_inner_timestep(i))

                    if self.logger and self.test:
                        self.logger.add_scalar(
                            'stock_level', self.stock_level[i], self.monitor_timestep[i]
                        )
                        self.logger.add_scalar(
                            'reward', self.reward_for_one_inner_timestep(i),
                            self.monitor_timestep[i]
                        )
                        self.logger.add_scalar(
                            'effective_action', 1 if j == 0 else 0, self.monitor_timestep[i]
                            # TODO: remove hack
                        )
                    self.monitor_timestep[i] += 1

            avg_rewards_all_materials.append(np.mean(rewards))

            # check to see if timesteps have reached the end.
            if self.monitor_timestep[i] >= len(self.history) - self.delivery_time[i]:
                done = True
                break
            else:
                done = False

        return avg_rewards_all_materials, done

    def define_new_state(self):

        new_state = [[0, 0] for _ in range(len(self.mat_info))]
        # loop over each material
        for i in range(len(self.stock_level)):
            if self.stock_level[i] >= 0:
                new_state[i][0] = self.stock_level[i]
            else:
                new_state[i][1] = abs(self.stock_level[i])

            # add demand history?
            new_state[i].extend(self.history.iloc[
                                self.monitor_timestep[i] - self.past_demand: self.monitor_timestep[
                                    i], i])

        return np.array(new_state)
        # return new_state

    def step(self, actions):
        self.action = actions  # TODO: remove hack
        rewards, done = self.update_stock_level_and_reward(actions)
        new_state = self.define_new_state()
        self.reward = rewards[0]
        self.time_step += 1
        return new_state, rewards[0], done, {}

    def reset(self):
        """
           Reset the state of the environment to an initial state readable by the policy network
           randomize if random_reset is true, else initialize as default.
           :return:
           """

        if self.random_reset:
            # initialize an empty Demand History (demand_history)
            self.monitor_timestep = [
                random.randint(self.past_demand, self.delivery_time[i] + self.past_demand)
                for i in range(len(self.mat_info))]
            self.demand_satisfaction = [0 for _ in range(len(self.mat_info))]

            # initialize current Stock Level (stock_level) : can be randomized
            self.stock_level = self.mat_info.loc[self.history.columns, 'Order_Volume'].values

        else:
            # initialize an empty Demand History (demand_history)
            self.monitor_timestep = [self.past_demand for _ in range(len(self.mat_info))]
            self.demand_satisfaction = [0 for _ in range(len(self.mat_info))]

            # initialize current Stock Level (stock_level) : can be randomized
            self.stock_level = self.mat_info.loc[self.history.columns, 'Order_Volume'].values

        return self.define_new_state()

    def render(self, param='none'):
        if self.logger and self.test:
            print(self.time_step, self.stock_level[0], self.action, self.reward)
            # self.logger.add_scalar(
            #     'stock_level', self.stock_level[0], self.time_step
            # )
            # self.logger.add_scalar(
            #     'effective_action', self.action, self.monitor_timestep[0]  # TODO: remove hack
            # )
            # self.logger.add_scalar(
            #     'reward', self.reward, self.time_step
            # )
