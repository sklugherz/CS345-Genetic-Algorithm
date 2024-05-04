import tensorflow as tf
import numpy as np
import yfinance as yf
import random


class Data:
    def __init__(self):
        self.data = yf.download('AAPL', period='5y', interval='1d')
        self.current_index = 0

    def get_current_price(self):
        if self.current_index < len(self.data):
            return self.data.iloc[self.current_index]['Close']
        else:
            return None

    def get_tensor(self):
        end_index = self.current_index + 1
        start_index = max(0, end_index - 15)
        return np.array(self.data.iloc[start_index:end_index])

    def next(self):
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
        else:
            self.current_index = 0

class Brain:
    def __init__(self):
        self.model = tf.keras.Sequential()
        initializer = tf.keras.initializers.RandomUniform(minval=-.5, maxval=.5, seed=42)
        # Define a model with 10 output neurons corresponding to binary classification tasks
        self.model.add(tf.keras.layers.Dense(80, activation='relu', input_shape=(75,), kernel_initializer=initializer))
        self.model.add(tf.keras.layers.Dense(10, activation='sigmoid', kernel_initializer=initializer))
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mse'])

    def prediction(self, inputs):

        inputs_np = np.array(inputs).reshape(1, -1)
        pred = self.model.predict(inputs_np)

        binary_outputs = (pred >= 0.5).astype(int)
        
        x, y = binary_outputs[0, 0], binary_outputs[0, 1]

        binary_string = ''.join(str(bit) for bit in binary_outputs[0, 2:])
        decimal_value = int(binary_string, 2)

        z = decimal_value / 255

        return [x, y, z]

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    
class Agent:

    data = Data()
    id = 0

    def __init__(self):
        self.id = Agent.id
        Agent.id = Agent.id + 1
        self.returns = 0
        self.fitness = 0
        self.input_dimension = 75
        self.row_dimension = 5
        self.row_data = Agent.data.get_tensor()[0] # numpy array of length 6

        print(self.row_data)
        temp = []
        for i in range(self.row_dimension):
            temp.append(self.row_data[i])
        for _ in range(self.input_dimension - self.row_dimension):
            temp.append(0)

        self.inputs = temp # python list of length 75
        self.current_price = self.inputs[0]
        self.brain = Brain()
        self.predictions = self.brain.prediction(self.inputs) # returns [{0, 1}, {0, 1}, float_between(0, 1)]
        self.active_trades = []

    def __lt__(self, other):
        if self.returns < other.returns:
            return True
        else:
            return False

    def make_trades(self):
        if self.predictions[0] > 0.5:
            if self.predictions[1] > 0.5:
                self.buy(self.current_price * (1 + self.predictions[2]))
            else:
                self.sell(self.current_price * (1 - self.predictions[2]))

    def buy(self, target):
        if target > self.inputs[0] * 1.10:
            self.fitness -= 10
        else:
            self.active_trades.append([target, 2*self.current_price - target, target - self.current_price, 1])

    def sell(self, target):
        if target < self.inputs[0] * 0.9:
            self.fitness -= 10
        else:
            self.active_trades.append([target, 2*self.current_price - target, self.current_price - target, 0])

    def close_trades(self):
        to_keep = []
        # print(f"Active Trades: {self.active_trades}")
        for trade in self.active_trades:
            if trade[3] == 1:
                if trade[0] <= self.current_price:
                    self.returns += trade[2]
                elif trade[1] >= self.current_price:
                    self.returns -= trade[2]
                else:
                    to_keep.append(trade)
            else:
                if trade[0] >= self.current_price:
                    self.returns += trade[2]
                elif trade[1] <= self.current_price:
                    self.returns -= trade[2]
                else:
                    to_keep.append(trade)
        self.active_trades = to_keep
        # print(f"Active Trades: {self.active_trades}")

    def force_close_trades(self):
        clear = []
        for trade in self.active_trades:
            open = (trade[0] + trade[1]) / 2
            if trade[3] == 1:
                self.returns += self.current_price - open
            else:
                self.returns += open - self.current_price
        self.active_trades = clear

    def get_next_inputs(self):
    # goes to next day, and appends the new data to inputs
        self.data.next()
        self.row_data = Agent.data.get_tensor()[0]
        for i in range(self.row_dimension):
            self.inputs.insert(0, self.row_data[i])
            self.inputs.pop()
        self.current_price = self.inputs[0]
        self.predictions = self.brain.prediction(self.inputs)

    def cycle(self):
        self.close_trades()
        self.make_trades()
        self.get_next_inputs()

# agent = Agent()
# for i in range(100):
#     agent.cycle()
# agent.force_close_trades()
# print(agent.current_price)
# print(agent.returns)

class Gen:
    def __init__(self, population_size, generations, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def crossover(self, parent1, parent2):
        child = Agent()
        p1_weights = parent1.brain.get_weights()
        p2_weights = parent2.brain.get_weights()
        new_weights = []
    
        for p1_layer, p2_layer in zip(p1_weights, p2_weights):
            if len(p1_layer.shape) == 1:  # This means the layer is one-dimensional (biases)
                gene_cutoff = np.random.randint(0, p1_layer.size)
                new_gene = np.concatenate([p1_layer[:gene_cutoff], p2_layer[gene_cutoff:]])
            else:  # This means the layer is two-dimensional (weights)
                gene_cutoff = np.random.randint(0, p1_layer.size)
                new_gene = np.concatenate([p1_layer[:gene_cutoff], p2_layer[gene_cutoff:]])
    
            new_weights.append(new_gene)

        child.brain.set_weights(new_weights)
        return child
    
    def mutate(self, agent):
        weights = agent.brain.get_weights()
        mutated_weights = []
        for weight in weights:
            if np.random.rand() < self.mutation_rate:
                mutation_matrix = np.random.uniform(-0.1, 0.1, weight.shape)
                weight += mutation_matrix
            mutated_weights.append(weight)
        agent.brain.set_weights(mutated_weights)

    def simulate(self):
        agents = []
        for _ in range(self.population_size):
            agent = Agent()
            agents.append(agent)
        for generation in range(self.generations):
            for agent in agents:
                for i in range(100):
                    agent.cycle()
                agent.force_close_trades()
                print(f"Agent {agent.id}'s returns after {i + 1} cycles: {agent.returns}")
            ranked_agents = sorted(agents)
            ranked_agents.reverse()
            print(f"=== Gen {generation} ===\n Best: ${ranked_agents[0].returns}. Worst: ${ranked_agents[-1].returns}")

            elites = ranked_agents[:len(ranked_agents) // 2]
            new_generation = []
            for _ in range(self.population_size):
                child = self.crossover(random.choice(elites), random.choice(elites))
                self.mutate(child)
                new_generation.append(child)

            agents = new_generation


test = Gen(20, 10)
test.simulate()