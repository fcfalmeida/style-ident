import numpy as np
from numpy.typing import ArrayLike


def l1_norm(vector: ArrayLike):
    """Computes the l1 (Manhattan) norm of a vector

    Args:
        vector: A vector or array of vectors.

    Returns:
        The l1 norm of the vector or an array of l1 norm value for each vector.
    """
    return np.sum(np.abs(vector), axis=1)


def l2_norm(vector: ArrayLike):
    """Computes the l2 (Euclidian) norm of a vectoor

    Args:
        vector: A vector or array of vectors.

    Returns:
        The l2 norm of the vector or an array of l2 norm value for each vector.
    """
    return np.sqrt(np.sum(np.power(vector, 2), axis=1))
