import pytest
import numpy as np
import src.utils.math as math


class TestMath:
    def test_l1_norm(self):
        vectors = np.array([[0.5, 0.3, -1.2, 2.3], [-1.5, 0, 0.3, -0.7]])

        expected = np.array([4.3, 2.5])

        result = math.l1_norm(vectors)

        assert np.array_equal(result, expected)

    def test_l2_norm(self):
        vectors = np.array([[0.5, 0.3, -1.2, 2.3], [-1.5, 0, 0.3, -0.7]])

        expected = np.array([2.65894716006, 1.68226038413])

        result = math.l2_norm(vectors)

        assert np.allclose(result, expected)

    @pytest.mark.parametrize('vector1, vector2, expected', [
        (
            np.array([[1, 2, 3, 1.5, 2.5], [4, 5, 6, 4.5, 5.5]]),
            np.array([[0, 1, 2, 1.2, 3], [0.2, 1.2, 1, 3, 2]]),
            np.array([17.3, 37.3])
        ),
        (
            np.array([[1 + 2j, 0.5 + 0.6j], [0.2 + 1.7j, 0.8 - 0.01j]]),
            np.array([[3.2 - 1.2j, -0.9 + 0.9j], [-0.2 - 5j, 2.2j + 0.07j]]),
            np.array([
                4.609999999999999 + 5.11j,
                8.482700000000001 + 0.47600000000000003j])
        ),
        (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            32
        )
    ])
    def test_dot_product(self, vector1, vector2, expected):
        result = math.dot_product(vector1, vector2)

        assert np.allclose(result, expected)

    @pytest.mark.parametrize('vector1, vector2, expected', [
        (
            np.array([0.5, 0.3, -1.2, 2.3]),
            np.array([-1.5, 0, 0.3, -0.7]),
            2.22444543
        ),
        (
            np.array([0.5 + 0.2j, 0.3 - 1.2j, -1.2 + 0.06j, 2.3 + 0j]),
            np.array([-1.5j, 0 + 0j, 0.3 - 1.01j, -0.7 - 0.3j]),
            0
        )
    ])
    def test_cosine_distance(self, vector1, vector2, expected):
        result = math.cosine_distance(vector1, vector2)

        assert np.isclose(result, expected)
