import numpy as np
from neural_network.utils import weight_init


class Layer():


    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, x):
        pass


class Linear(Layer):


    def __init__(self, input_size, output_size):

        super().__init__()

        self.W, self.b = weight_init(input_size, output_size)
        self.dW, self.db = 0, 0
        self.input = None
        self.dx = None

    
    def forward(self, x):
        """
        x: output of the previous layer
        """
        
        self.input = x
        out = self.W @ x + self.b
        return out

    def backward(self, dZ):
        """
        dZ: del_L / del_Z, where Z is the output
        """
        self.dW = dZ @ self.input.T
        self.db = dZ
        self.dx = self.W.T @ dZ

        return self.dx


class Sigmoid(Layer):

    
    def __init__(self):     
        
        self.out = None
        super().__init__()

    def forward(self, x):
        """
        x: output of the previous layer
        """

        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dZ):
        """
        dZ: del_L / del_Z, where Z is the output
        """

        return dZ @ self.out @ (1 - self.out)
