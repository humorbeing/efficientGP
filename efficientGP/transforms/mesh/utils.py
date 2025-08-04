import torch
import numpy as np


def pack(padded, num):
    return torch.cat([t[:n] for t, n in zip(padded, num)], dim=0)


def pad_x(x, n_padded=65000):
    return np.pad(x, [[0, n_padded - len(x)], [0, 0]])

def pad_xs(xs, n_padded=65000):
    return np.stack([pad_x(x, n_padded) for x in xs])


def retrieve_data(values, keys, key, dtype=np.float32):
    return np.array(values[:, keys.index(key)], dtype=dtype)