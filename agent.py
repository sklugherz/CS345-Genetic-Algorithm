import random
import numpy as np
import brain
from data import data
from input_vector import *


class agent:

    weights_length
    HISTORY_VAR = 15
    INPUT_VECTOR_SIZE = 5
    next_id = 0

    def __init__(self, weights=None):
        

        #  initialize weights for initial_population
        if weights is None:
            self.weights = np.array([])
            for _ in self.brain.get_weights():
                self.weights.append(random.uniform(-0.1, 0.1))
        else:
            self.weights = weights
        
        #  initialize neural network
        self.brain = brain(weights)

        #  initialize id
        self.id = agent.next_id
        agent.next_id = agent.next_id + 1

        #  initialize trades list
        self.active_trades = []

        #  track returns for fitness
        self.returns = 0
        self.num_trades = 0

        #  initialize input vector
        self.input_v = input_vector()
        self.input_v.initialize(agent.HISTORY_VAR, agent.INPUT_VECTOR_SIZE)



    def update_trades(self, current_price):
        trades_to_close = []
        for trade in self.active_trades:
            if trade[3] == 1:
                #  close long position if applicable
                if trade[1] <= current_price:
                    self.earnings += trade[1] - trade[0]
                    trades_to_close.append(trade)
                elif trade[2] >= current_price:
                    self.earnings += trade[2] - trade[0]
                    trades_to_close.append(trade)
            else:
                #  close short position if applicable
                if trade[1] <= current_price:
                    self.earnings += trade[1] - trade[0]
                    trades_to_close.append(trade)
                elif trade[2] >= current_price:
                    self.earnings += trade[2] - trade[0]
                    trades_to_close.append(trade)
        
        self.active_trades = self.active_trades - trades_to_close


    def input(self, input_vector):
        self.input_v.input(input_vector)


    def run_brain(self):
        pass


    def make_trades(self):
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
    
    # def update(self):
    #     if len(self.active_trades) == 0:
    #         return
    #     else:
    #         for i in range(len(self.active_trades) - 1, -1, -1):
    #             if self.active_trades[i][3] == 1:
    #                 if self.current_price >= self.active_trades[i][1]:
    #                     self.total_earnings += self.active_trades[i][1] - self.active_trades[i][0]
    #                     self.active_trades.pop(i)
    #             else:
    #                 if self.current_price <= self.active_trades[i][1]:
    #                     self.total_earnings += self.active_trades[i][0] - self.active_trades[i][1]
    #                     self.active_trades.pop(i)