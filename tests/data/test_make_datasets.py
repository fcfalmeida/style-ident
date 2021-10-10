import pytest
import pandas as pd


class TestMakeDatasets:
    @pytest.fixture
    def data(scope="class"):
        df = pd.DataFrame(
            [
                {
                    "piece": "bach_2.mp3",
                    "time": 0,
                    "c1": 0.03,
                    "c2": 0.07,
                    "c3": 0.11,
                    "c4": 0.09,
                    "c5": 0.01,
                    "c6": 0.04,
                    "c7": 0.06,
                    "c8": 0.13,
                    "c9": 0.06,
                    "c10": 0.21,
                    "c11": 0.12,
                    "c12": 0.07,
                },
                {
                    "piece": "bach_2.mp3",
                    "time": 0.1,
                    "c1": 0.02,
                    "c2": 0.08,
                    "c3": 0.14,
                    "c4": 0.05,
                    "c5": 0.01,
                    "c6": 0.15,
                    "c7": 0.10,
                    "c8": 0.05,
                    "c9": 0.03,
                    "c10": 0.07,
                    "c11": 0.18,
                    "c12": 0.12,
                },
                {
                    "piece": "bach_2.mp3",
                    "time": 0.2,
                    "c1": 0,
                    "c2": 0,
                    "c3": 0,
                    "c4": 0,
                    "c5": 0,
                    "c6": 0,
                    "c7": 0,
                    "c8": 0,
                    "c9": 0,
                    "c10": 0,
                    "c11": 0,
                    "c12": 0,
                },
            ]
        )

        df["time"] = pd.to_timedelta(df["time"], unit="s")
        df = df.set_index(["piece", "time"])

        return df

    def test_main(self, data):
        pass
