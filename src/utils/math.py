import numpy as np
from numpy.typing import ArrayLike

def l1_norm(vector: ArrayLike):
    return np.sum(np.abs(vector), axis=1)

def l2_norm(vector: ArrayLike):
    return np.sqrt(np.sum(np.power(vector, 2), axis=1))