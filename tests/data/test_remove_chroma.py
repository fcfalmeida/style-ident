from src.data.remove_chroma import RemoveChroma
import pytest
import numpy as np
import pandas as pd


class TestRemoveChroma:
    @pytest.fixture(scope="class")
    def data(self):
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
                    "ic1": 0.2,
                    "ic2": 0.08,
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
                    "ic1": 0.12,
                    "ic2": 0.03,
                },
            ]
        )

        df["time"] = pd.to_timedelta(df["time"], unit="s")
        df = df.set_index(["piece", "time"])

        return df

    def test_remove_chroma(self, data):
        expected = pd.DataFrame(
            [
                {"piece": "bach_2.mp3", "time": 0, "ic1": 0.2, "ic2": 0.08},
                {"piece": "bach_2.mp3", "time": 0.1, "ic1": 0.12, "ic2": 0.03},
            ]
        )
        expected["time"] = pd.to_timedelta(expected["time"], unit="s")
        expected = expected.set_index(["piece", "time"])

        result = RemoveChroma().run(data)

        compare = pd.DataFrame(
            np.isclose(expected, result), columns=expected.columns)

        assert compare.all(axis=None)
