import numpy as np
import src.utils.math as math

class TestMath:
    def test_l1_norm(self):
        vectors = np.array([
            [0.5, 0.3, -1.2, 2.3],
            [-1.5, 0, 0.3, -0.7]
        ])

        expected = np.array([4.3, 2.5])

        result = math.l1_norm(vectors)

        assert np.array_equal(result, expected)

    def test_l2_norm(self):
        vectors = np.array([
            [0.5, 0.3, -1.2, 2.3],
            [-1.5, 0, 0.3, -0.7]
        ])

        expected = np.array([2.65894716006, 1.68226038413])

        result = math.l2_norm(vectors)

        print(result)

        assert np.allclose(result, expected)