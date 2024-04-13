import numpy as np
from collections import deque

from neural_network.layer import Layer, Loss


class NeuralNetwork:
    """
    Neural network class
    Assumes that the computational graph is in a lattice form (only one root, with no parents, and one end node, with no children)
    """
    def __init__(self, root_layer: Layer, output_layer: Layer, loss_layer: Loss = None):
        self.root_layer = root_layer

        if isinstance(output_layer, Layer):
            self.output_layer = [output_layer]
        else:
            self.output_layer = output_layer # assuming it's already a list of layers

        self.loss_layer = loss_layer

        self.n_layers = self._initial_dfs(self.root_layer, set())
        self._forward_order = self._topological_sort_forward()
        self._backward_order = self._topological_sort_backward()

    def forward(self, x: np.ndarray):
        self.root_layer.forward(x)
        for layer in self._forward_order[1:]:
            if not isinstance(layer, Loss): # regular layer
                x_node = np.concatenate([parent.out for parent in layer.parents], axis=0)
                layer.forward(x_node)


        if len(self.output_layer) == 1:
            return self.output_layer[0].out
        else:
            return [layer.out for layer in self.output_layer]


    def backward(self):
        for layer in self._backward_order:
            layer.backward()

    def step(self, lr=1e-3):
        for layer in self._backward_order:
            layer.update(lr)

    def _topological_sort_forward(self):
        def _topological_sort_rec(layer: Layer, visited: set, stack: deque):
            visited.add(layer)

            for child in layer.children:
                if child not in visited:
                    _topological_sort_rec(child, visited, stack)

            stack.append(layer)
    
        stack = deque()
        visited = set()

        _topological_sort_rec(self.root_layer, visited, stack)

        order = []

        while len(stack) > 0:
            order.append(stack.pop())

        return order
    
    def _topological_sort_backward(self):
        def _topological_sort_rec(layer: Layer, visited: set, stack: deque):
            visited.add(layer)

            for parent in layer.parents:
                if parent not in visited:
                    _topological_sort_rec(parent, visited, stack)

            stack.append(layer)
    
        stack = deque()
        visited = set()

        _topological_sort_rec(self.loss_layer if isinstance(self.loss_layer, Loss) else self.output_layer[0], visited, stack)

        order = []

        while len(stack) > 0:
            order.append(stack.pop())

        return order

    def _initial_dfs(self, layer: Layer, visited: set):
        child_count = 0
        visited.add(layer)

        for child in layer.children:
            if child not in visited:
                child_count += self._initial_dfs(child, visited)
        return 1 + child_count
        