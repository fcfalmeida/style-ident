import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import distance


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


def cosine_distance(vector1: ArrayLike, vector2: ArrayLike) -> float:
    """Computes the cosine distance between two vectors or arrays of vectors.

    Args:
        vector1: A vector or array of vectors.
        vector2: Another vector or array of vectors.

    Returns:
        Cosine distance between `vector1` and `vector2` (angle).
    """

    # If the dot product is a complex number, retrieve only its real part
    dot = dot_product(vector1, vector2).real

    try:
        norm1 = np.linalg.norm(vector1, axis=1)
        norm2 = np.linalg.norm(vector2, axis=1)
    except np.AxisError:
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

    # dist = np.arccos(dot / (norm1 * norm2))
    norm_prod = norm1 * norm2
    dist = np.arccos(
        np.divide(dot, norm_prod, where=norm_prod != 0)
    )

    # Possible divison by zero above will return NaN
    return np.nan_to_num(dist)

    # return np.arccos(1 - distance.cosine(vector1, vector2))
