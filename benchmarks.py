import pandas as pd
import matplotlib.pyplot as plt
from rlib.shared.utils import Logger

import csv

def write_csv(data):
    with open('output_test/total_rewards.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

class SimulateBenchmark:

    def __init__(self, demand_history, mat_info, thresholds):

        self.demand_history = demand_history
        self.mat_info = mat_info
        self.thresholds = thresholds

        self.reorder = mat_info.loc[demand_history.columns, 'Order_Volume'].values
        self.inv_costs = mat_info.loc[demand_history.columns, 'Inventory_Costs'].values
        self.stock_out_costs = mat_info.loc[demand_history.columns, 'Stock_Out_Costs'].values
        self.delivery_time = mat_info.loc[demand_history.columns, 'Delivery_Time_days'].values

        self.demand_satisfaction = [0 for _ in range(len(mat_info))]
        self.action_flag = [0 for _ in range(len(mat_info))]
        self.pending_actions = [[0] * self.delivery_time[i] for i in range(len(mat_info))]
        self.stock_level = mat_info.loc[demand_history.columns, 'Current_Stock_Level'].values

        self.stock_history = [[] for _ in range(len(self.mat_info))]
        self.action_history = [[0] * self.delivery_time[i] for i in range(len(mat_info))]
        self.reward_history = [[] for _ in range(len(self.mat_info))]
        self.inventory_reward_history = [[] for _ in range(len(self.mat_info))]
        self.stock_out_reward_history = [[] for _ in range(len(self.mat_info))]

    def register_action(self):

        for i in range(len(self.stock_level)):

            if self.stock_level[i] <= self.thresholds[i] and self.action_flag[i] == 0:

                self.action_flag[i] = 1
                self.pending_actions[i].append(1)
                self.action_history[i].append(1)
            else:
                self.pending_actions[i].append(0)
                self.action_history[i].append(0)

    def update_stock_level(self, current_demand):
        """
        accept daily action and daily demand and update stock level.
        """

        for i in range(len(self.stock_level)):
            self.demand_satisfaction[i] = 0
            # update for actions
            if self.pending_actions[i][0] == 1:

                self.action_flag[i] = 0
                if self.stock_level[i] < 0 and self.stock_level[i] + self.reorder[i] > 0:
                    self.demand_satisfaction[i] = abs(self.stock_level[i])
                    self.stock_level[i] = self.stock_level[i] + self.reorder[i]
                elif self.stock_level[i] < 0 and self.stock_level[i] + self.reorder[i] < 0:
                    self.demand_satisfaction[i] = abs(self.reorder[i])
                    self.stock_level[i] = self.stock_level[i] + self.reorder[i]
                else:
                    self.stock_level[i] = self.stock_level[i] + self.reorder[i]
            self.pending_actions[i].pop(0)
            assert len(self.pending_actions[i]) == self.delivery_time[i]

            # update for demand
            if self.stock_level[i] > 0:
                self.demand_satisfaction[i] += min(current_demand[i], self.stock_level[i])
            self.stock_level[i] = self.stock_level[i] - current_demand[i]

            self.stock_history[i].append(self.stock_level[i])

    def reward_function(self):

        for i in range(len(self.stock_level)):
            inventory_reward = - self.inv_costs[i] * max(self.stock_level[i], 0)
            stock_out_reward = self.stock_out_costs[i] * min(self.stock_level[i], 0)
            reward = inventory_reward + stock_out_reward

            self.reward_history[i].append(reward)
            self.inventory_reward_history[i].append(inventory_reward)
            self.stock_out_reward_history[i].append(stock_out_reward)

    def simulate(self):

        for i in range(len(self.demand_history)):
            current_demand = list(self.demand_history.loc[i])

            self.register_action()
            self.update_stock_level(current_demand)
            self.reward_function()

        return self.stock_history, self.action_history, self.reward_history, self.inventory_reward_history, self.stock_out_reward_history

    
if __name__ == '__main__':
    threshold = {'B120': 12593,
                 'B120BP': 34371,
                 'Q115': 45616,
                 'Q120': 145850,
                 'Q2100H': 1931,
                 'Q3015': 1020,
                 'TA2J6500': 1481}

    for mood in ['train', 'test']:
        output_dir_logger = f'output_test/benchmarks_{mood}'
        logger = Logger(path=output_dir_logger, comment=None, verbosity='DEBUG', experiment_name=f'benchmark_{mood}')
        for material in ['Q115', 'B120BP', 'B120', 'Q120', 'TA2J6500', 'Q2100H', 'Q3015']:
            total_reward = []
            # q115_test = pd.read_csv(f"Data/Preprocessing/{mood}_q115.csv")
            # mat_info_q115 = pd.read_csv("Data/Material_Information_q115.csv", sep=";", index_col="Material")
            mat_info = pd.read_csv("Data/Material_Information.csv", sep=";", index_col="Material")
            hist_data = pd.read_csv(f"Data/Preprocessing/{mood}.csv")
            mat_info = mat_info.loc[[material]]
            hist_data = hist_data[[material]]

            Sebastians_model = SimulateBenchmark(hist_data, mat_info, [threshold[material]]) #hack
            stock_history, action_history, reward_history, inventory_reward_history, stock_out_reward_history = Sebastians_model.simulate()

            for i in range(hist_data.size):
                logger.add_scalar(
                    f'stock_level_{material}', stock_history[0][i], i
                )
                logger.add_scalar(
                    f'effective_action_{material}', action_history[0][i], i
                )
                logger.add_scalar(
                    f'reward_{material}', reward_history[0][i], i
                )
                logger.add_scalar(
                    f'demand_{material}', hist_data.iloc[i, 0], i
                )
            print(material, mood, sum(reward_history[0]))
            write_csv([f'benchmarks_{mood}', material, sum(reward_history[0]), sum(stock_out_reward_history[0]), sum(inventory_reward_history[0])])

        logger.close()