import pytest
import pandas as pd
import numpy as np
from src.features.complexity import Complexity
from src.data.constants import CHROMA_COLS

class TestComplexity:
    @pytest.fixture
    def data(scope='class'):
        df = pd.DataFrame([
            {'piece': 'bach_2.mp3', 'time': 0, 'c1': 0.21, 'c2': 0.45, 'c3': 0.77, 'c4': 0.09, 
                'c5': 0.23, 'c6': 0.50, 'c7': 0.14, 'c8': 0.08, 'c9': 0.91, 'c10': 0.83, 'c11': 0.77, 'c12': 0.11},
            {'piece': 'bach_2.mp3', 'time': 0.1, 'c1': 0.03, 'c2': 0.06, 'c3': 0.44, 'c4': 0.31, 
                'c5': 0.97, 'c6': 0.13, 'c7': 0.19, 'c8': 0.25, 'c9': 0.37, 'c10': 0.52, 'c11': 0.81, 'c12': 0.01},
        ])

        df['time'] = pd.to_timedelta(df['time'], unit='s')
        df = df.set_index(['piece', 'time'])

        return df

    def test_sort_chroma_fifths(self, data):
        expected = np.array([
            [0.21, 0.08, 0.77, 0.83, 0.23, 0.11, 0.14, 0.45, 0.91, 0.09, 0.77, 0.50],
            [0.03, 0.25, 0.44, 0.52, 0.97, 0.01, 0.19, 0.06, 0.37, 0.31, 0.81, 0.13]
        ])

        result = Complexity()._sort_chroma_fifths(data[CHROMA_COLS].values)

        assert np.array_equal(result, expected)

    def test_sum_chroma_diff(self, data):
        expected = np.array([-0.9, -0.77])

        result = Complexity()._sum_chroma_diff(data[CHROMA_COLS].values)

        assert np.allclose(result, expected)

    def test_chroma_std(self, data):
        expected = np.array([-0.06746974977, -0.009995874579])

        result = Complexity()._chroma_std(data[CHROMA_COLS].values)

        print(result)

        assert(np.allclose(result, expected))