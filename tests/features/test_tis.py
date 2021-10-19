import pytest
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from TIVlib import TIVCollection
from src.features.tis import TIS


class TestTIS:
    @pytest.fixture
    def chroma_data(scope="class"):
        chroma = np.array(
            [
                [
                    1.20,
                    5.10,
                    0.90,
                    1.60,
                    2.50,
                    3.30,
                    6.00,
                    0.20,
                    0.30,
                    0.10,
                    0.02,
                    0.01,
                ],
                [
                    2.10,
                    1.50,
                    9.40,
                    6.10,
                    5.20,
                    2.20,
                    3.00,
                    0.40,
                    0.20,
                    0.50,
                    0.07,
                    0.04,
                ],
                [
                    0.30,
                    1.10,
                    4.90,
                    2.30,
                    2.40,
                    0.25,
                    5.37,
                    0.52,
                    0.13,
                    0.16,
                    2.02,
                    1.01,
                ],
                [
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                ],
                [
                    0.02,
                    0.01,
                    1.95,
                    1.69,
                    5.23,
                    4.35,
                    2.62,
                    0.26,
                    0.39,
                    0.10,
                    0.02,
                    0.01,
                ],
            ]
        )

        return chroma

    @pytest.fixture()
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
                    "hcdf_peak_idx": 0,
                    "hcdf_peak_mag": 0
                },
                {
                    "piece": "bach_2.mp3",
                    "time": 0.7,
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
                    "hcdf_peak_idx": 10,
                    "hcdf_peak_mag": 0.8
                },
                {
                    "piece": "bach_2.mp3",
                    "time": 1.2,
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
                    "hcdf_peak_idx": 13,
                    "hcdf_peak_mag": 1.3
                },
                {
                    "piece": "bach_2.mp3",
                    "time": 1.6,
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
                    "hcdf_peak_idx": 18,
                    "hcdf_peak_mag": 0.65
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
                    "hcdf_peak_idx": 0,
                    "hcdf_peak_mag": 0
                },
                {
                    "piece": "chopin_3.mp3",
                    "time": 0.4,
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
                    "hcdf_peak_idx": 7,
                    "hcdf_peak_mag": 1.24
                },
                {
                    "piece": "chopin_3.mp3",
                    "time": 0.9,
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
                    "hcdf_peak_idx": 15,
                    "hcdf_peak_mag": 0.93
                },
                {
                    "piece": "chopin_3.mp3",
                    "time": 1.3,
                    "c1": 1.05,
                    "c2": 3.01,
                    "c3": 1.93,
                    "c4": 1.69,
                    "c5": 0.35,
                    "c6": 0.42,
                    "c7": 1.02,
                    "c8": 2.31,
                    "c9": 0.02,
                    "c10": 0.17,
                    "c11": 2.41,
                    "c12": 0.01,
                    "hcdf_peak_idx": 21,
                    "hcdf_peak_mag": 1.79
                },
            ]
        )

        df = df.set_index(['piece', 'time'])

        return df

    def test_dissonance(self, chroma_data: ArrayLike):
        tivs = TIVCollection.from_pcp(chroma_data.T)

        expected = np.array(
            [
                0.7016734443427521,
                0.7960692272326073,
                0.7137129952692088,
                1.0,
                0.8020384634735304,
            ]
        )

        result = TIS()._dissonance(tivs)

        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "distance_type, expected",
        [
            (
                TIS.DIST_COSINE,
                np.array([1.0988581, 0.88569352, 0.47083198, 0, 1.16067857]),
            ),
            (
                TIS.DIST_EUCLIDEAN,
                np.array(
                    [
                        7.024411157159626,
                        4.205619975303846,
                        4.4070228830168,
                        3.9649816028134306,
                        5.148215450137535,
                    ]
                ),
            ),
        ],
    )
    def test_tonal_dispersion(
        self, chroma_data: ArrayLike, distance_type: int, expected: ArrayLike
    ):
        tivs = TIVCollection.from_pcp(chroma_data.T)

        result = TIS()._tonal_dispersion(chroma_data, tivs, distance_type)

        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "distance_type, expected",
        [
            (TIS.DIST_COSINE, np.array([1.9128754, 0.92266401, 0, 0, 0])),
            (
                TIS.DIST_EUCLIDEAN,
                np.array(
                    [
                        10.928152516237695,
                        6.080080353530224,
                        7.556838197613272,
                        5.225397157960991,
                        0,
                    ]
                ),
            ),
        ],
    )
    def test_distance(
        self, chroma_data: ArrayLike, distance_type: int, expected: ArrayLike
    ):
        tivs = TIVCollection.from_pcp(chroma_data.T)

        result = TIS()._distance(tivs, distance_type)

        assert np.allclose(result, expected)

    def test_tiv_mags(self, chroma_data: ArrayLike):
        tivs = TIVCollection.from_pcp(chroma_data.T)

        expected = np.array(
            [
                [
                    1.31919532,
                    3.00185009,
                    3.95403167,
                    2.79932772,
                    5.26701495,
                    0.21549694,
                ],
                [
                    1.91872448,
                    2.00210756,
                    2.05708604,
                    2.24760545,
                    2.63093143,
                    2.25415174,
                ],
                [
                    0.98709979,
                    0.32976971,
                    5.47476922,
                    1.44933394,
                    3.33113828,
                    3.58504399,
                ],
                [0, 0, 0, 0, 0, 0],
                [
                    2.2846165,
                    2.92844477,
                    1.87284191,
                    1.7715249,
                    1.97946427,
                    1.71621622
                ],
            ]
        )

        result = TIS()._tiv_mags(tivs)

        assert np.allclose(result, expected)

    @pytest.mark.parametrize('k, expected', [
        (
            0,
            np.array([
                0.43973177479452447,
                0.6395748270512249,
                0.32903326238757197,
                0.0,
                0.7615388318543962
            ])
        ),
        (
            5,
            np.array([
                0.028732925105982073,
                0.30055356561380653,
                0.47800586510263926,
                0.0,
                0.22882882882882896
            ])
        ),
        (
            4,
            np.array([
                0.36324241052334316,
                0.18144354706452584,
                0.22973367448713475,
                0.0,
                0.136514777052585
            ])
        )
    ])
    def test_descriptor(
        self, chroma_data: ArrayLike, k: int, expected: ArrayLike
    ):
        tivs = TIVCollection.from_pcp(chroma_data.T)

        tiv_mags = TIS()._tiv_mags(tivs)

        result = TIS()._descriptor(tiv_mags, k)

        assert np.allclose(result, expected)

    def test_hcdf_peak_interval(self):
        hcdf_peak_indexes = np.array([0, 7, 20, 38, 39])

        expected = np.array([7, 13, 18, 1, 0])

        result = TIS()._hcdf_peak_interval(hcdf_peak_indexes)

        assert np.allclose(result, expected)

    def test_run(self, data: pd.DataFrame):
        expected = pd.DataFrame([
            {
                "piece": "bach_2.mp3",
                "time": 0,
                "dissonance": 0.7269596,
                "chromaticity": 0.40453365,
                "dyadicity": 0.28422334,
                "triadicity": 0.44779999,
                "dim_quality": 0.1889306,
                "diatonicity": 0.21732042,
                "wholetoneness": 0.12074303,
                "cos_tonal_disp": 0.92781653,
                "euc_tonal_disp": 5.99694061,
                "cos_dist": 1.62210025,
                "euc_dist": 8.25962446,
                "hcdf_peak_interval": 10.0
            },
            {
                "piece": "bach_2.mp3",
                "time": 0.7,
                "dissonance": 0.86051336,
                "chromaticity": 0.40014579,
                "dyadicity": 0.16741983,
                "triadicity": 0.17265286,
                "dim_quality": 0.04029938,
                "diatonicity": 0.13624077,
                "wholetoneness": 0.19373219,
                "cos_tonal_disp": 0.83445766,
                "euc_tonal_disp": 2.73590814,
                "cos_dist": 0,
                "euc_dist": 3.6818924,
                "hcdf_peak_interval": 3.0
            },
            {
                "piece": "bach_2.mp3",
                "time": 1.2,
                "dissonance": 1,
                "chromaticity": 0,
                "dyadicity": 0,
                "triadicity": 0,
                "dim_quality": 0,
                "diatonicity": 0,
                "wholetoneness": 0,
                "cos_tonal_disp": 0,
                "euc_tonal_disp": 2.68011437,
                "cos_dist": 0,
                "euc_dist": 7.79135677,
                "hcdf_peak_interval": 5.0
            },
            {
                "piece": "bach_2.mp3",
                "time": 1.6,
                "dissonance": 0.70482838,
                "chromaticity": 0.44148486,
                "dyadicity": 0.16930013,
                "triadicity": 0.10934247,
                "dim_quality": 0.44405399,
                "diatonicity": 0.22995932,
                "wholetoneness": 0.03006012,
                "cos_tonal_disp": 1.52347408,
                "euc_tonal_disp": 8.11866062,
                "cos_dist": 0,
                "euc_dist": 0,
                "hcdf_peak_interval": 0.0
            },
            {
                "piece": "chopin_3.mp3",
                "time": 0,
                "dissonance": 0.67099447,
                "chromaticity": 0.48787723,
                "dyadicity": 0.19559162,
                "triadicity": 0.45034618,
                "dim_quality": 0.33863814,
                "diatonicity": 0.25832167,
                "wholetoneness": 0.2724359,
                "cos_tonal_disp": 0.51068575,
                "euc_tonal_disp": 5.49711035,
                "cos_dist": 1.59634496,
                "euc_dist": 9.51889966,
                "hcdf_peak_interval": 7.0
            },
            {
                "piece": "chopin_3.mp3",
                "time": 0.4,
                "dissonance": 0.86051336,
                "chromaticity": 0.40014579,
                "dyadicity": 0.16741983,
                "triadicity": 0.17265286,
                "dim_quality": 0.04029938,
                "diatonicity": 0.13624077,
                "wholetoneness": 0.19373219,
                "cos_tonal_disp": 1.18997034,
                "euc_tonal_disp": 4.36515155,
                "cos_dist": 1.83722253,
                "euc_dist": 8.82507864,
                "hcdf_peak_interval": 8.0
            },
            {
                "piece": "chopin_3.mp3",
                "time": 0.9,
                "dissonance": 0.73066732,
                "chromaticity": 0.35815171,
                "dyadicity": 0.26995724,
                "triadicity": 0.40360278,
                "dim_quality": 0.25771809,
                "diatonicity": 0.19792183,
                "wholetoneness": 0.00304569,
                "cos_tonal_disp": 0.91610324,
                "euc_tonal_disp": 5.64469452,
                "cos_dist": 1.17674747,
                "euc_dist": 7.91044942,
                "hcdf_peak_interval": 6.0
            },
            {
                "piece": "chopin_3.mp3",
                "time": 1.3,
                "dissonance": 0.7293748,
                "chromaticity": 0.20753578,
                "dyadicity": 0.25073002,
                "triadicity": 0.27527971,
                "dim_quality": 0.35473626,
                "diatonicity": 0.19622033,
                "wholetoneness": 0.05767894,
                "cos_tonal_disp": 0.46834717,
                "euc_tonal_disp": 3.95545254,
                "cos_dist": 0,
                "euc_dist": 0,
                "hcdf_peak_interval": 0.0
            }
        ])

        expected = expected.set_index(["piece", "time"])

        result = TIS().run(data)

        compare = pd.DataFrame(
            np.isclose(expected, result), columns=expected.columns)

        print(compare)

        assert compare.all(axis=None)
