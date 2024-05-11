import numpy as np
import stats
import matplotlib.pyplot as plt
import seaborn as sns


def weight_init(input_size, output_size, init_type="he"):

    if init_type == "he":

        std = np.sqrt(2 / (input_size + output_size))
        weight = np.random.normal(0, std, (output_size, input_size))
        bias = np.random.normal(0, std, (output_size, 1))

    return weight, bias


def one_hot_vector(x, length=4):

    vec = np.zeros((length, 1))
    vec[x] = 1
    return vec

