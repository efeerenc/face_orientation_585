import numpy as np
from neural_network.utils import weight_init
import neural_network.convutils as convutils

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

    def init_weights(self):
        pass


class Loss(Layer):
    def __init__(self, parents=None, layer_id=None):
        super().__init__(parents, layer_id=layer_id)


class Linear(Layer):

    def __init__(self, input_size, output_size, parents=None, layer_id=None):

        super().__init__([parents], layer_id=layer_id)

        self.input_size = input_size
        self.output_size = output_size
        self.init_weights()
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

    def init_weights(self):
        self.W, self.b = weight_init(self.input_size, self.output_size)

class Conv2D(Layer):

    def __init__(
        self,
        kernel_shape,
        pad=1,
        stride=1,
        parents=None,
        layer_id=None,
    ):
        super().__init__([parents], layer_id=layer_id)
        self.kernel_shape = kernel_shape
        self.num_filters = kernel_shape.shape[0]
        self.in_channels = kernel_shape.shape[1]
        self.kernel_size = kernel_shape.shape[2] # assuming square kernels
        self.pad = pad
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        self.W, self.b = convutils.kernel_init(self.kernel_shape)

    def forward(self, x):
        """
            Performs a forward convolution.

            Parameters:
            - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
            Returns:
            - out: previous layer convolved.
        """
        batch_size, in_channels, in_height, in_width = x.shape

        out_channels = self.num_filters
        out_height = ((in_height + 2 * self.pad - self.kernel_size) // self.stride) + 1
        out_width = ((in_width + 2 * self.pad - self.kernel_size) // self.stride) + 1

        X_col = convutils.im2col(x, self.kernel_size, self.kernel_size, self.stride, self.pad)
        w_col = self.W.reshape((self.num_filters, -1))
        b_col = self.b.reshape(-1, 1)
        # Perform matrix multiplication.
        out = w_col @ X_col + b_col
        # Reshape back matrix to image.
        self.out = np.array(np.hsplit(out, batch_size)).reshape((batch_size, out_channels, out_height, out_width))

        self.cache = x, X_col, w_col # reduces memory allocations later
        return self.out

    def backward(self):
        """
            Distributes error from previous layer to convolutional layer and
            compute error for the current convolutional layer.

            Parameters:
            - dout: error from previous layer.

            Returns:
            - dX: error of the current convolutional layer.
            - self.W['grad']: weights gradient.
            - self.b['grad']: bias gradient.
        """

        dZ = 0
        for child in self.children:
            dZ += child.grad

        x, x_col, w_col = self.cache
        batch_size, _, _, _ = x.shape
        #print("conv x shape", x.shape)
        # Compute bias gradient.
        self.db = np.sum(dZ, axis=(0,2,3))
        # Reshape dout properly.
        dZ = dZ.reshape(dZ.shape[0] * dZ.shape[1], dZ.shape[2] * dZ.shape[3])
        dZ = np.array(np.vsplit(dZ, batch_size))
        dZ = np.concatenate(dZ, axis=-1)
        #print("conv dZ", dZ.shape)
        # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
        dX_col = w_col.T @ dZ
        #print("conv dX_col", dX_col.shape)
        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        dw_col = dZ @ x_col.T
        # Reshape back to image (col2im).
        self.grad = convutils.col2im(dX_col, x.shape, self.kernel_size, self.kernel_size, self.stride, self.pad)
        #print("conv grad", self.grad.shape)
        # Reshape dw_col into dw.
        self.dW = dw_col.reshape((dw_col.shape[0], self.num_filters, self.kernel_size, self.kernel_size))

        return self.grad 
    
    def update(self, lr=1e-3):
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db

    def __str__(self):
        return f"Conv2D{'' if self.id==None else ' ' + str(self.id)}: {(self.num_filters, self.in_channels, self.kernel_size, self.kernel_size)}, pad={self.pad}, stride={self.stride}"

class MaxPool2D(Layer):
    def __init__(self, window_size: int, pad: int = 0, stride: int = 2, parents=None, layer_id=None):
        super().__init__([parents], layer_id=layer_id)
        self.window_size = window_size
        self.pad = pad
        self.stride = stride

    def forward(self, x: np.ndarray):
        """
        Apply max pooling.

        Parameters:
        - X: Output of activation function.

        Returns:
        - A_pool: X after average pooling layer. 
        """
        self.cache = x # better to hold on, reduces memory allocations

        batch_size, in_channels, in_height, in_width = x.shape
        out_channels = in_channels
        out_height = ((in_height + 2 * self.pad - self.window_size) // self.stride) + 1
        out_width = ((in_width + 2 * self.pad - self.window_size) // self.stride) + 1

        X_col = convutils.im2col(x, self.window_size, self.window_size, self.stride, self.pad) 
        X_col = X_col.reshape(out_channels, X_col.shape[0] // out_channels, -1)
        A_pool = np.max(X_col, axis=1)
        # Reshape A_pool properly.
        A_pool = np.array(np.hsplit(A_pool, batch_size))
        A_pool = A_pool.reshape(batch_size, out_channels, out_height, out_width)
        self.out = A_pool
        return self.out
    
    def backward(self):
        """
            Distributes error through pooling layer.

            Returns:
            - dX: Conv layer updated with error.
        """
        dZ = 0
        for child in self.children:
            dZ += child.grad

        x = self.cache

        batch_size, in_channels, in_height, in_width = x.shape

        out_channels = in_channels
        out_height = ((in_height + 2 * self.pad - self.window_size) // self.stride) + 1
        out_width = ((in_width + 2 * self.pad - self.window_size) // self.stride) + 1

        dout_flatten = dZ.reshape(in_channels, -1) / (self.window_size * self.window_size)
        dZ_col = np.repeat(dout_flatten, self.window_size*self.window_size, axis=0)
        dZ = convutils.col2im(dZ_col, x.shape, self.window_size, self.window_size, self.stride, self.pad)
        # Reshape dX properly.
        dZ = dZ.reshape(batch_size, -1)
        dZ = np.array(np.hsplit(dZ, in_channels))
        dZ = dZ.reshape(batch_size, in_channels, in_height, in_width)
        self.grad = dZ
        return self.grad

    def __str__(self):
        return f"MaxPool2D{'' if self.id==None else ' ' + str(self.id)}: {(self.window_size, self.window_size)}, pad={self.pad}, stride={self.stride}"

class Flatten(Layer):
    def __init__(self, parents=None, layer_id=None):
        super().__init__([parents], layer_id=layer_id)

    def forward(self, x):
        self.batch_size, self.in_channels, self.in_height, self.in_width = x.shape

        self.out = x.reshape((-1, 1))
        return self.out
    
    def backward(self):
        dZ = 0
        for child in self.children:
            dZ += child.grad

        self.grad = dZ.reshape((self.batch_size, self.in_channels, self.in_height, self.in_width))
        return self.grad

    def __str__(self):
        return f"Flatten{'' if self.id==None else ' ' + str(self.id)}"

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
            #print(child, child.grad.shape)
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
