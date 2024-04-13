import numpy as np
from neural_network.layer import *
from neural_network.net import *
from queue import Queue
import matplotlib.pyplot as plt

"""
x_input = np.ones((10, 1))
y_target = np.array([[0.5], [0.5]])
linear1 = Linear(10, 5, layer_id=1)
relu1 = ReLU(linear1, layer_id=1)
linear2 = Linear(5, 2, relu1, layer_id=2)
relu2 = ReLU(linear2, layer_id=2)

linear3 = Linear(2, 2, linear2, layer_id=3)
relu3 = ReLU(linear3, layer_id=3)

addition2 = Addition([relu3, relu2], layer_id=2)

relu4 = ReLU(addition2, layer_id=4)

addition1 = Addition([relu2, relu4], layer_id=1)
sigmoid = Sigmoid(addition1, layer_id=1)
loss_layer = MSE_Loss(sigmoid, layer_id=1)
"""

x_input = np.ones((10, 1))
y_target = np.array([[0.5], [0.5]])
linear1 = Linear(10, 5)
relu1 = ReLU(linear1)
linear2 = Linear(5, 2, relu1)
relu2 = ReLU(linear2)
linear3 = Linear(2, 2, relu2)
sigmoid = Sigmoid(linear3)
loss_layer = MSE_Loss(sigmoid)

model = NeuralNetwork(linear1, sigmoid, loss_layer)

print(model.n_layers)
print(model.output_layer)

print("Forward order:")
for layer in model._forward_order:
    print(layer)

print("\nBackward order:")
for layer in model._backward_order:
    print(layer)

net_out= model.forward(x_input)

print("\nNet out:", net_out)


loss_array = []
for i in range(10000):
    out = model.forward(x_input)
    print(out)
    loss = model.loss_layer.forward(out, y_target)
    loss_array.append(loss)
    model.backward()
    model.step()