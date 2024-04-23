import random
import numpy as np
import brain
from data import data


class agent:

    agents = []
    INPUT_DIM = 5
    next_id = 0
    TICKER = 'AAPL'
    data = data(TICKER, '1y', '1d')

    def __init__(self, fitness, total_earnings, risk_reward, start_datetime):

        self.fitness = fitness
        self.total_earnings = total_earnings
        self.risk_reward = risk_reward
        self.start_datetime = start_datetime

        self.num_trades = 0
        self.num_spawn = 0

        self.current_time = 1
        self.current_price = data[self.current_time][3]

        self.id = agent.next_id
        agent.next_id += 1

        self.brain = brain(agent.INPUT_DIM)

        self.active_trades = []

    def mutate():
        pass

    def reproduce():
        pass

    #  buy order denoted by 1
    def buy(self, current_price, target_price):
        if target_price < current_price:
            self.fitness -= 10
        else:
            stop_loss = 2*current_price - target_price
            buy_vector = np.array([current_price, target_price, stop_loss, 1])
            self.active_trades.append(buy_vector)

    #  sell orders denoted by 0
    def sell(self, open_price, target_price):
        if target_price > open_price:
            self.fitness -= 10
        else:
            stop_loss = 2*open_price - target_price
            sell_vector = np.array([open_price, target_price, stop_loss, 0])
            self.active_trades.append(sell_vector)

    def set_fitness(self):
        if self.num_trades == 0:
            self.fitness = 0
        else:
            self.fitness = self.total_earnings + self.total_earnings/self.num_trades**2 #  an attempt to discourage a higher number of trades


    def crossover(parent1, parent2):
        pass

    def choose_parents(agents):
        parent_1_index = random.randint(0, len(agents) // 2)
        parent_2_index = random.randint(0, len(agents) // 2)

        parent_1 = agents[parent_1_index]
        parent_2 = agents[parent_2_index]

        return parent_1, parent_2
    
    def update(self):
        if len(self.active_trades) == 0:
            return
        else:
            for i in range(len(self.active_trades) - 1, -1, -1):
                if self.active_trades[i][3] == 1:
                    if self.current_price >= self.active_trades[i][1]:
                        self.total_earnings += self.active_trades[i][2] - self.active_trades[i][1]
                        self.active_trades.pop(i)
                else:
                    if self.current_price <= self.active_trades[i][1]:
                        self.total_earnings += self.active_trades[i][1] - self.active_trades[i][2]
                        self.active_trades.pop(i)

        self.current_time += 1
        self.num_trades += 1