import numpy as np


def PCA(data, first_k=90, shape=(120, 128)):

    center = data.mean(axis=0)
    centered = data - center  # Subtract mean

    u, s, vt = np.linalg.svd(centered)

    lambdas = np.square(s) / (data.shape[0] - 1)

    first_k = 90
    eig_first_k = lambdas[0:first_k] / sum(lambdas)
    preserved = [sum(eig_first_k[0 : i + 1]) for i in range(first_k)]
    eig_idx = first_k - np.sum(np.array(preserved) >= 0.9)

    return center, lambdas, vt, eig_idx
