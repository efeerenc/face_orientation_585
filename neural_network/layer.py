import numpy as np
from neural_network.utils import weight_init

import random


class Layer:

    def __init__(self, parents=[], layer_id=None):

        if any(elem is None for elem in parents):
            parents = []

        self.parents = parents
        self.children = []

        self.id = layer_id
        self.input = None
        self.out = None

        for parent in parents:
            parent.children.append(self)

    def forward(self, x):

        if self.parents:
            node_input = []
            for parent in self.parents:
                node_input.append(parent.out)
            node_input = np.concatenate(node_input, axis=0)
        else:
            node_input = x

        return node_input

    def backward(self):
        pass

    def update(self, lr):
        pass


class Loss(Layer):
    def __init__(self, parents=None, layer_id=None):
        super().__init__(parents, layer_id=layer_id)


class Linear(Layer):

    def __init__(self, input_size, output_size, parents=None, layer_id=None):

        super().__init__([parents], layer_id=layer_id)

        self.input_size = input_size
        self.output_size = output_size
        self.W, self.b = weight_init(input_size, output_size)
        self.dW, self.db = 0, 0
        self.input = None
        self.dx = None
        self.out = None

    def forward(self, x):
        """
        x: output of the previous layer
        """
        x = super().forward(x)
        self.input = x
        self.out = self.W @ x + self.b
        return self.out

    def backward(self):
        """
        dZ: del_L / del_Z, where Z is the output
        """
        dZ = 0
        for child in self.children:
            dZ += child.grad

        self.dW = dZ @ self.input.T
        self.db = dZ
        self.dx = self.W.T @ dZ
        self.grad = self.dx

        return self.grad

    def update(self, lr):
        """
        Update parameters after self.backward() w.r.t. given learning rate (lr)
        """
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db

    def __str__(self):
        return f"Linear{'' if self.id==None else ' ' + str(self.id)}: ({self.input_size}, 1) -> ({self.output_size}, 1)"


class Conv2d(Layer):

    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        parents=None,
        stride=1,
        padding=0,
        dilation=1,
        layer_id=None,
    ):

        super().__init__([parents], layer_id=layer_id)

        pass


class Sigmoid(Layer):

    def __init__(self, parents=None, layer_id=None):

        self.out = None
        super().__init__([parents], layer_id=layer_id)

    def forward(self, x):
        """
        x: output of the previous layer
        """
        x = super().forward(x)
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self):
        """
        dZ: del_L / del_Z, where Z is the output
        """
        dZ = 0
        for child in self.children:
            dZ += child.grad

        self.grad = dZ @ self.out.T @ (1 - self.out)
        return self.grad

    def update(self, lr):
        pass

    def __str__(self) -> str:
        return f"Sigmoid{'' if self.id==None else ' ' + str(self.id)}"


class ReLU(Layer):

    def __init__(self, parents=None, layer_id=None):

        self.out = None
        super().__init__([parents], layer_id=layer_id)

    def forward(self, x):
        """
        x: output of the previous layer
        """
        x = super().forward(x)
        self.mask = x >= 0
        self.out = x * self.mask
        return self.out

    def backward(self):
        """
        dZ: del_L / del_Z, where Z is the output
        """
        dZ = 0
        for child in self.children:
            dZ += child.grad

        self.grad = dZ * self.mask
        return self.grad

    def update(self, lr):
        pass

    def __str__(self) -> str:
        return f"ReLU{'' if self.id==None else ' ' + str(self.id)}"


class Softmax(Layer):

    def __init__(self, parents=None, layer_id=None):

        self.out = None
        super().__init__([parents], layer_id=layer_id)

    def forward(self, x):
        """
        x: output of the previous layer
        """
        exp_result = np.exp(x)
        self.out = exp_result / np.sum(exp_result)
        return self.out

    def backward(self):
        """
        dZ: del_L / del_Z, where Z is the output
        """
        dZ = 0
        for child in self.children:
            dZ += child.grad

        grad = self.out * np.eye(len(self.out)) - (self.out @ self.out.T)
        self.grad = grad @ dZ
        return self.grad

    def update(self, lr):
        pass

    def __str__(self) -> str:
        return f"Softmax{'' if self.id==None else ' ' + str(self.id)}"


class Addition(Layer):

    def __init__(self, parents=None, layer_id=None):

        super().__init__(parents, layer_id=layer_id)

        self.layer1 = parents[0]
        self.layer2 = parents[1]
        self.out = None

    def forward(self, x):
        """
        x: output of the previous layer
        """
        x = super().forward(x)
        self.out = self.layer1.out + self.layer2.out
        return self.out

    def backward(self):
        """
        dZ: del_L / del_Z, where Z is the output
        """
        dZ = 0
        for child in self.children:
            dZ += child.grad

        self.grad = dZ
        return self.grad

    def update(self, lr):
        """
        Update parameters after self.backward() w.r.t. given learning rate (lr)
        """
        return

    def __str__(self) -> str:
        return f"Addition{'' if self.id==None else ' ' + str(self.id)}"


class MSE_Loss(Loss):

    def __init__(self, parents=None, layer_id=None):

        super().__init__([parents], layer_id=layer_id)

    def forward(self, y_pred: np.ndarray, target: np.ndarray):
        """
        y_tuple = (y_pred, y_target)
        """

        self.y_pred = y_pred
        self.y_target = target

        self.out = (1 / target.shape[0]) * np.linalg.norm(
            self.y_pred - self.y_target
        ) ** 2
        return self.out

    def backward(self):

        dZ = 1
        # print(self.y_pred)
        # print(self.y_target)
        self.grad = np.array(self.y_pred - self.y_target).reshape(self.y_pred.shape)
        return -self.grad

    def update(self, lr):
        pass

    def __str__(self) -> str:
        return f"MSE Loss{'' if self.id==None else ' ' + str(self.id)}"


class CrossEntropy(Loss):

    def __init__(self, parents=None, layer_id=None):

        super().__init__([parents], layer_id=layer_id)

    def forward(self, y_pred: np.ndarray, target: np.ndarray):
        """
        y_tuple = (y_pred, y_target)
        """

        self.y_pred = y_pred
        self.y_target = target

        self.out = np.where(self.y_target == 1, -np.log(self.y_pred), 0).sum(axis=0)

        return self.out

    def backward(self):

        dZ = 1
        # print(self.y_pred)
        # print(self.y_target)
        # self.grad = np.array(self.y_pred - self.y_target).reshape(self.y_pred.shape)
        self.grad = np.where(self.y_target == 1, -1 / self.y_pred, 0)
        return -self.grad

    def update(self, lr):
        pass

    def __str__(self) -> str:
        return f"CrossEntropy{'' if self.id==None else ' ' + str(self.id)}"
