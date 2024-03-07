import numpy as np

def weight_init(input_size, output_size, init_type="he"):

    weight = np.random.normal(0, 1, (output_size, input_size))
    bias = np.random.normal(0, 1, (output_size, 1))

    return weight, bias