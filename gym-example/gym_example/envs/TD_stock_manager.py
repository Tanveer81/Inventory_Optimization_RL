import math
import random
from abc import ABC
import numpy as np
import gym

class StockManager(gym.Env, ABC):

    def __init__(self, hist_data, mat_info, random_reset, sinusoidal_demand=False,
                 demand_satisfaction=False, past_demand=3, sine_type=3, noisy_demand=False):

        super(StockManager, self).__init__()
        self.noisy_demand = noisy_demand
        self.sine_type = sine_type
        self.past_demand = past_demand
        self.demand_sat = demand_satisfaction
        self.sinusoidal_demand = sinusoidal_demand
        self.history = hist_data
        self.mat_info = mat_info
        self.random_reset = random_reset

        self.reorder = mat_info.loc[hist_data.columns, 'Order_Volume'].values
        self.inv_costs = mat_info.loc[hist_data.columns, 'Inventory_Costs'].values
        self.stock_out_costs = mat_info.loc[hist_data.columns, 'Stock_Out_Costs'].values
        self.delivery_time = mat_info.loc[hist_data.columns, 'Delivery_Time_days'].values
        self.storage_cap = mat_info.loc[hist_data.columns, 'Order_Volume'].values * 4

        # initialize an empty Demand History (demand_history)
        self.current_demand = [0 for _ in range(len(mat_info))]
        self.demand_satisfaction = [0 for _ in range(len(mat_info))]
        self.demand_history = np.zeros((len(mat_info), self.past_demand), int)

        # initialize an empty Pending Action History (pending_actions)
        self.effective_action = [0 for _ in range(len(mat_info))]
        self.pending_actions = [[] for _ in range(len(mat_info))]
        self.days_to_next_reorder = [0 for _ in range(len(mat_info))]

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
        list1 = [self.storage_cap[0], self.storage_cap[0], 1]
        list2 = [self.storage_cap[0]] * self.past_demand
        self.observation_space = gym.spaces.Box(low=np.array([0]*(self.past_demand + 3)),
                                                high=np.array(list1+list2),
                                                dtype=np.float32)

        # define action space (currently for only one material)
        self.action_space = gym.spaces.Discrete(2)

    def observe_new_action(self, actions):
        """
        Register action proposed by policy
        Maintain memory of the last time reorder (action) was requested.

        :param actions: policy, decides whether to place reorder or not.
        :return: None
        """

        for i in range(len(actions)):

            if actions[i] == 1:
                # if self.stock_level[i] > (0.85 * self.storage_cap[i]):
                #     continue
                self.pending_actions[i].append(self.delivery_time[i])

    
    def get_sinusoidal_demand(self, timestep, sine_type=3):
        if sine_type==1:
            dmax = 60000
            return [math.floor(dmax/2 * math.sin(6*math.pi*timestep) + dmax/2 + random.randint(0, 50000))]
        elif sine_type==2:
            dmax = 40000
            return [random.choice([max(math.floor(dmax/3 * math.sin(6*math.pi*timestep)
                                       + dmax/3 * math.sin(24*math.pi*timestep)
                                       + dmax/3 * math.sin(183*math.pi*timestep)
                                       + dmax/2 + random.randint(0, 40000)*random.randint(0, 1)), 0), 0])]
        elif sine_type==3:
            dmax = 5000
            return [random.choice([max(math.floor(dmax / 2 * math.sin(6 * math.pi * timestep)
                                                 + dmax / 2 * math.sin(24 * math.pi * timestep)
                                                 + dmax / 2 * math.sin(180 * math.pi * timestep)
                                                 + 3 * dmax / 4
                                                 + random.randint(8000, 9000) * random.randint(0, 1)
                                                 + random.randint(10000, 12000) * random.randint(0, 1) * random.randint(0, 1)
                                                 + random.randint(15000, 20000) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1)
                                                 + random.randint(30000, 35000) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1)
                                                 + random.randint(45000, 50000) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1)), 0), 0])]


    def observe_new_demand(self, time_step):
        """
        Register daily demand
        Maintain demand history of past 7 days
        :TODO: add noise
        :param time_step: current time step in the real world.
        :return: None
        """
        if self.sinusoidal_demand:
            self.current_demand = self.get_sinusoidal_demand(time_step, self.sine_type)
        else:
            if self.noisy_demand:
                self.current_demand = list(self.history.loc[time_step] + random.randint(100, 9000) * random.randint(0, 1))
            else:
                self.current_demand = list(self.history.loc[time_step])
        # pop the first column (oldest history)
        self.demand_history = self.demand_history[:, 1:]
        # add the new demand at the end (latest history)
        self.demand_history = np.c_[self.demand_history, self.current_demand]

    def update_stock_level(self):
        """
        Register all reorders and demands that have occurred.
        decrement any existing pending action counters.

        :return: None
        """
        # register reorders:
        # recursive loop for each material
        for i in range(len(self.pending_actions)):

            self.effective_action[i] = 0
            # loop over all pending actions for the selected material
            self.demand_satisfaction[i] = 0
            for j in range(len(self.pending_actions[i])):
                self.pending_actions[i][j] -= 1
                # check if the action can finally take effect
                if self.pending_actions[i][j] == 0:
                    self.effective_action[i] = 1
                    if self.stock_level[i] < 0 and self.stock_level[i] + self.reorder[i] > 0:
                        self.demand_satisfaction[i] = abs(self.stock_level[i])
                        self.stock_level[i] = self.stock_level[i] + self.reorder[i]
                    elif self.stock_level[i] < 0 and self.stock_level[i] + self.reorder[i] < 0:
                        self.demand_satisfaction[i] = abs(self.reorder[i])
                        self.stock_level[i] = self.stock_level[i] + self.reorder[i]
                    else:
                        self.stock_level[i] = self.stock_level[i] + self.reorder[i]

            self.pending_actions[i] = [x for x in self.pending_actions[i] if x != 0]
            self.days_to_next_reorder[i] = self.pending_actions[i][0]/self.delivery_time[i] \
                if len(self.pending_actions[i])!= 0 \
                else 0

        # register demand: today's demand
        for i in range(len(self.current_demand)):
            if self.stock_level[i] > 0:
                self.demand_satisfaction[i] += min(self.current_demand[i], self.stock_level[i])
            self.stock_level[i] = self.stock_level[i] - self.current_demand[i]

    def define_new_state(self):
        """
        :return: new_state
        """
        new_state = [[0, 0, self.days_to_next_reorder[i]] for i in range(len(self.mat_info))]
        for i in range(len(self.stock_level)):
            if self.stock_level[i] >= 0:
                new_state[i][0] = self.stock_level[i]
            else:
                new_state[i][1] = abs(self.stock_level[i])

            new_state[i] = new_state[i] + list(self.demand_history[i])

        return np.array(new_state)

    def define_new_reward(self):
        """
        calculate reward of current time step based on stock level.
        :return:
        """
        rewards = []
        for i in range(len(self.stock_level)):
            demand_sat = self.demand_satisfaction[i] if self.demand_sat else 0
            if self.stock_level[i] >= 0:
                rewards.append(demand_sat - self.inv_costs[i] * self.stock_level[i])
            else:
                rewards.append(demand_sat - self.stock_out_costs[i] * abs(self.stock_level[i]))

        return rewards

    def step(self, actions, time_step):
        """
        Execute one time step within the environment
        :param self:
        :param actions: action vector defined by the policy network
        :param time_step: current time step in the real world
        :return:
        new_state:
        rewards:
        done:
        """

        self.observe_new_action(actions)
        self.observe_new_demand(time_step)
        self.update_stock_level()

        rewards = self.define_new_reward()
        new_state = self.define_new_state()

        return new_state, rewards, {}

    def reset(self):
        """
        Reset the state of the environment to an initial state readable by the policy network
        randomize if random_reset is true, else initialize as default.
        :return:
        """

        if self.random_reset:
            self.stock_level = None
            self.demand_history = None
            self.pending_actions = None

        else:
            self.demand_history = np.zeros((len(self.mat_info), self.past_demand), int)
            self.pending_actions = [[] for _ in range(len(self.mat_info))]
            self.days_to_next_reorder = [0 for _ in range(len(self.mat_info))]
            self.stock_level = self.mat_info.loc[self.history.columns, 'Order_Volume'].values

        return self.define_new_state()
