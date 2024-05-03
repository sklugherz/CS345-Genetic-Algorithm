import numpy as np


class input_vector:

    def __init__(self):
        self.vector = []
        self.numpy_vector = np.array(self.vector)

    def input(self, input):
        for i in range(len(input)):
            self.vector.insert(0, input[i])
            self.vector.pop()

    def initialize(self, history_var, input_vector_size):
        for i in range(history_var * input_vector_size):
            self.vector[i] == None