import pytest
import numpy as np
from numpy.typing import ArrayLike
from TIVlib import TIVCollection
from src.features.tis import TIS


class TestTIS:
    @pytest.fixture
    def data(self):
        chroma = np.array([
            [1.2, 5.1, 0.9, 1.6, 2.5, 3.3, 6.0, 0.2, 0.3, 0.1, 0.02, 0.01],
            [2.1, 1.5, 9.4, 6.1, 5.2, 2.2, 3.0, 0.4, 0.2, 0.5, 0.07, 0.04],
            [0.3, 1.1, 4.9, 2.3, 2.4, 0.25,
                5.37, 0.52, 0.13, 0.16, 2.02, 1.01],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.02, 0.01, 1.95, 1.69, 5.23,
                4.35, 2.62, 0.26, 0.39, 0.1, 0.02, 0.01]
        ])

        return TIVCollection.from_pcp(chroma.T)

    def test_dissonance(self, data: ArrayLike):
        expected = np.array([
            0.7016734443427521,
            0.7960692272326073,
            0.7137129952692088,
            1.0,
            0.8020384634735304
        ])

        result = TIS()._dissonance(data)

        print(result)

        assert np.allclose(result, expected)
