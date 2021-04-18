import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
from rlib.shared.utils import Logger

def visualize(stocks, actions, rewards):

    plt.plot(rewards[0])
    plt.title("Reward for model")
    plt.xlabel("time steps")
    plt.ylabel("reward")
    plt.show()

    effective_actions = actions[0][0:-12]
    markers_on = []
    for i in range(len(effective_actions)):
        if effective_actions[i] == 1:
            markers_on.append(i)

    plt.plot(stocks[0], '-d', markevery=markers_on)
    plt.title("Stock Levels for Sebastian's model")
    plt.xlabel("time steps")
    plt.ylabel("stock level")
    plt.show()


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
            reward = - self.inv_costs[i] * max(self.stock_level[i], 0) + \
                     self.stock_out_costs[i] * min(self.stock_level[i], 0) # + self.demand_satisfaction[i]

            self.reward_history[i].append(reward)

    def simulate(self):

        for i in range(len(self.demand_history)):
            current_demand = list(self.demand_history.loc[i])

            self.register_action()
            self.update_stock_level(current_demand)
            self.reward_function()

        return self.stock_history, self.action_history, self.reward_history

def visualize(agents):
    actions = []
    rewards = []
    stocks = []
    for agent in agents:
        try:
            with open(f'../output/{agent}/test_actions.txt') as f:
                actions.append([float(line.rstrip()) for line in f])
            with open(f'../output/{agent}/test_stats.txt') as f:
                rewards.append([float(line.rstrip()) for line in f])
            with open(f'../output/{agent}/test_storages.txt') as f:
                stocks.append([float(line.rstrip()) for line in f])
        except:
            print(f'{agent} results not present')
            agents.remove(agent)
    for reward in rewards:
        plt.plot(reward)
    plt.legend(agents, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Reward for model")
    plt.xlabel("time steps")
    plt.ylabel("reward")
    plt.show()

    markers_on = []
    for action in actions:
        effective_actions = action[0:-12]
        markers = []
        for i in range(len(effective_actions)):
            if effective_actions[i] == 1:
                markers.append(i)
        markers_on.append(markers)

    for stock, marker in zip(stocks, markers_on):
        plt.plot(stock, '-d', markevery=marker)
    plt.legend(agents, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Stock Levels for Sebastian's model")
    plt.xlabel("time steps")
    plt.ylabel("stock level")
    plt.show()

    
if __name__ == '__main__':
    for material in ['B120BP', 'B120', 'Q120', 'TA2J6500', 'Q115', 'Q2100H', 'Q3015']:
        print(material)
        for mood in ['test', 'train']:
            output_dir_logger = f'output/benchmarks_{mood}'
            logger = Logger(path=output_dir_logger, comment=None, verbosity='DEBUG', experiment_name=f'benchmark_{mood}')
            # q115_test = pd.read_csv(f"Data/Preprocessing/{mood}_q115.csv")
            # mat_info_q115 = pd.read_csv("Data/Material_Information_q115.csv", sep=";", index_col="Material")

            mat_info = pd.read_csv("Data/Material_Information.csv", sep=";", index_col="Material")
            mat_info = mat_info.loc[[material]]
            hist_data = pd.read_csv(f"Data/Preprocessing/{mood}.csv")
            hist_data = hist_data[[material]]

            threshold = {'B120': 12593,
                        'B120BP': 34371,
                        'Q115': 45616,
                        'Q120': 145850,
                        'Q2100H': 1931,
                        'Q3015': 1020,
                        'TA2J6500': 1481}

            Sebastians_model = SimulateBenchmark(hist_data, mat_info, [threshold[material]]) #hack
            stock_history, action_history, reward_history = Sebastians_model.simulate()

            for i in range(hist_data.size):
                logger.add_scalar(
                    f'stock_level_{material}', stock_history[0][i], i
                )
                logger.add_scalar(
                    f'effective_action_{material}', action_history[0][i], i
                    # TODO: remove hack
                )
                logger.add_scalar(
                    f'reward_{material}', reward_history[0][i], i
                )
                logger.add_scalar(
                    f'demand_{material}', hist_data.iloc[i, 0], i
                )

        print()
    #     visualize(stock_history, action_history, reward_history)
    #
    #     output_dir = Path('../output/benchmark')
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     np.savetxt(f"{output_dir}/test_stats.txt", reward_history[0], delimiter=',')
    #     np.savetxt(f"{output_dir}/test_actions.txt", action_history[0], delimiter=',')
    #     np.savetxt(f"{output_dir}/test_storages.txt", stock_history[0], delimiter=',')

    # agents = next(os.walk('../output'))[1]
    # # visualize(['benchmark'])
    # visualize(['benchmark', 'q_full', 'q_sine_past_demand_7', 'q_sine_3x', 'q_full_demand_satisfaction','q_sine_complex_model'])