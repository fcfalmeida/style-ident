import numpy as np
from numpy.typing import ArrayLike


def l1_norm(vector: ArrayLike):
    """Computes the l1 (Manhattan) norm of a vector.

    Args:
        vector: A vector or array of vectors.

    Returns:
        The l1 norm of the vector or an array of l1 norm value for each vector.
    """
    return np.sum(np.abs(vector), axis=1)


def l2_norm(vector: ArrayLike):
    """Computes the l2 (Euclidean) norm of a vector.

    Args:
        vector: A vector or array of vectors.

    Returns:
        The l2 norm of the vector or an array of l2 norm value for each vector.
    """
    return np.sqrt(np.sum(np.power(vector, 2), axis=1))


def dot_product(vector1: ArrayLike, vector2: ArrayLike) -> ArrayLike:
    """Computes the dot product between two vectors.
    If both vectors are 2-dimensional, it computes the dot product between
    each row of `vector1` and `vector2`.
    Args:
        vector1: A vector or array of vectors.
        vector2: Another vector or array of vectors.
    Returns:
        Dot product between `vector1` and `vector2.
    """
    try:
        return np.sum(vector1 * vector2, axis=1)
    except np.AxisError:
        return np.sum(vector1 * vector2)


def complex_cosine_dist(vector1, vector2):
    with np.errstate(divide='ignore', invalid='ignore'):
        vector1_split = np.concatenate(
            (vector1.real, vector1.imag), axis=1
        )
        vector2_split = np.concatenate(
            (vector2.real, vector2.imag), axis=1
        )

        vector1_norms = np.linalg.norm(vector1, axis=1)
        vector2_norms = np.linalg.norm(vector2, axis=1)
        dot_prod = dot_product(vector1_split, vector2_split)

        dist = np.arccos(dot_prod / (vector1_norms * vector2_norms))

        return np.nan_to_num(dist)


def normalize(values: ArrayLike):
    sum = np.sum(values, axis=1)

    return np.divide(
        values,
        sum[:, None],
        out=np.zeros_like(values),
        where=sum[:, None] != 0
    )


def entropy(values: ArrayLike):
    """Computes the Shannon entropy of a 2D array of values, row-wise.

    Args:
        values: A 2-dimensional vector of values

    Returns:
        An array of the entropy value for each row of `values´
    """
    return np.sum(-values * np.log2(values, where=values > 0), axis=1)
