import pytest
import pandas as pd


class TestHCDFSegmentation:
    @pytest.fixture
    def data(scope="class"):
        df = pd.DataFrame(
            [
                {
                    "piece": "bach_2.mp3",
                    "time": 0,
                    "c1": 0.22,
                    "c2": 0.34,
                    "c3": 1.09,
                    "c4": 1.21,
                    "c5": 0.22,
                    "c6": 0.96,
                    "c7": 1.91,
                    "c8": 2.85,
                    "c9": 0.02,
                    "c10": 0.03,
                    "c11": 0.80,
                    "c12": 0.04,
                },
                {
                    "piece": "bach_2.mp3",
                    "time": 0.1,
                    "c1": 0.11,
                    "c2": 2.33,
                    "c3": 1.01,
                    "c4": 0.99,
                    "c5": 0.09,
                    "c6": 0.48,
                    "c7": 1.49,
                    "c8": 1.77,
                    "c9": 2.41,
                    "c10": 4.27,
                    "c11": 3.38,
                    "c12": 2.73,
                },
                {
                    "piece": "bach_2.mp3",
                    "time": 0.2,
                    "c1": 0.00,
                    "c2": 0.00,
                    "c3": 0.00,
                    "c4": 0.00,
                    "c5": 0.00,
                    "c6": 0.00,
                    "c7": 0.00,
                    "c8": 0.00,
                    "c9": 0.00,
                    "c10": 0.00,
                    "c11": 0.00,
                    "c12": 0.00,
                },
                {
                    "piece": "bach_2.mp3",
                    "time": 0.3,
                    "c1": 0.00,
                    "c2": 0.12,
                    "c3": 0.70,
                    "c4": 0.39,
                    "c5": 0.22,
                    "c6": 1.28,
                    "c7": 0.37,
                    "c8": 0.60,
                    "c9": 1.11,
                    "c10": 0.13,
                    "c11": 0.02,
                    "c12": 0.05,
                },
                {
                    "piece": "chopin_3.mp3",
                    "time": 0,
                    "c1": 0.02,
                    "c2": 0.14,
                    "c3": 0.07,
                    "c4": 0.91,
                    "c5": 0.23,
                    "c6": 0.69,
                    "c7": 1.11,
                    "c8": 2.12,
                    "c9": 0.03,
                    "c10": 0.04,
                    "c11": 0.81,
                    "c12": 0.07,
                },
                {
                    "piece": "chopin_3.mp3",
                    "time": 0.1,
                    "c1": 0.11,
                    "c2": 2.33,
                    "c3": 1.01,
                    "c4": 0.99,
                    "c5": 0.09,
                    "c6": 0.48,
                    "c7": 1.49,
                    "c8": 1.77,
                    "c9": 2.41,
                    "c10": 4.27,
                    "c11": 3.38,
                    "c12": 2.73,
                },
                {
                    "piece": "chopin_3.mp3",
                    "time": 0.2,
                    "c1": 0.99,
                    "c2": 0.11,
                    "c3": 1.29,
                    "c4": 1.86,
                    "c5": 0.43,
                    "c6": 0.54,
                    "c7": 1.92,
                    "c8": 2.31,
                    "c9": 0.08,
                    "c10": 0.07,
                    "c11": 0.23,
                    "c12": 0.02,
                },
            ]
        )

        return df
