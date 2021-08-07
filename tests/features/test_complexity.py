import pytest
import pandas as pd
import numpy as np
from src.features.complexity import Complexity
from src.data.constants import CHROMA_COLS

class TestComplexity:
    @pytest.fixture
    def data(scope='class'):
        df = pd.DataFrame([
            {'piece': 'bach_2.mp3', 'time': 0, 'c1': 0.03, 'c2': 0.07, 'c3': 0.11, 'c4': 0.09, 
                'c5': 0.01, 'c6': 0.04, 'c7': 0.06, 'c8': 0.13, 'c9': 0.06, 'c10': 0.21, 'c11': 0.12, 'c12': 0.07},
            {'piece': 'bach_2.mp3', 'time': 0.1, 'c1': 0.02, 'c2': 0.08, 'c3': 0.14, 'c4': 0.05, 
                'c5': 0.01, 'c6': 0.15, 'c7': 0.10, 'c8': 0.05, 'c9': 0.03, 'c10': 0.07, 'c11': 0.18, 'c12': 0.12}
        ])

        df['time'] = pd.to_timedelta(df['time'], unit='s')
        df = df.set_index(['piece', 'time'])

        return df

    def test_sort_chroma_fifths(self, data):
        expected = np.array([
            [0.03, 0.13, 0.11, 0.21, 0.01, 0.07, 0.06, 0.07, 0.06, 0.09, 0.12, 0.04],
            [0.02, 0.05, 0.14, 0.07, 0.01, 0.12, 0.1, 0.08, 0.03, 0.05, 0.18, 0.15]
        ])

        result = Complexity()._sort_chroma_fifths(data[CHROMA_COLS].values)

        assert np.array_equal(result, expected)

    def test_sum_chroma_diff(self, data):
        expected = np.array([0.65, 0.59])

        result = Complexity()._sum_chroma_diff(data[CHROMA_COLS].values)

        assert np.allclose(result, expected)

    def test_chroma_std(self, data):
        expected = np.array([0.82148763, 0.81760848])

        result = Complexity()._chroma_std(data[CHROMA_COLS].values)

        assert np.allclose(result, expected)

    def test_neg_slope(self, data):
        expected = np.array([1.1631701538, 1.1326878205])

        result = Complexity()._neg_slope(data[CHROMA_COLS].values)

        assert np.allclose(result, expected)

    def test_entropy(self, data):
        expected = np.array([0.92430871434, 0.91369479299])

        result = Complexity()._entropy(data[CHROMA_COLS].values)

        assert np.allclose(result, expected)

        # Flat chroma vector entropy should be 1
        flat_vector = np.array([[1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 
            1/12, 1/12, 1/12, 1/12, 1/12, 1/12]])
        result = Complexity()._entropy(flat_vector)

        assert np.allclose(result, np.array([1]))

        # Sparse chroma vector entropy should be 0
        sparse_vector = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
        result = Complexity()._entropy(sparse_vector)

        assert np.allclose(result, np.array([0]))


