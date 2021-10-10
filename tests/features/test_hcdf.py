import pytest
import pandas as pd
import numpy as np
from src.features.hcdf import HCDF
from src.data.constants import CHROMA_COLS


class TestHCDF:
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
                    "piece": "vivaldi_1.mp3",
                    "time": 0,
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
                    "piece": "vivaldi_1.mp3",
                    "time": 0.1,
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
                    "piece": "grieg_4.mp3",
                    "time": 0,
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

    def test_calc_tivs(self, data: pd.DataFrame):
        expected = np.array(
            [
                [
                    -1.18745113 - 0.25057351j,
                    1.58513932 - 1.63016547j,
                    -3.96388029 + 3.2874097j,
                    0.3250774 - 2.81525286j,
                    3.15059582 + 0.05888653j,
                    -0.90557276 + 0.0j,
                ],
                [
                    0.28469108 + 1.16619076j,
                    -1.31054131 - 0.27633859j,
                    -1.78561254 - 0.86823362j,
                    -0.17094017 - 0.57981758j,
                    -1.97500974 + 0.04361075j,
                    -1.45299145 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    -1.29613433 - 0.27242565j,
                    -0.24048096 - 1.33288078j,
                    0.55310621 - 1.12925852j,
                    -3.48697395 + 5.67515645j,
                    2.34180358 - 2.37365678j,
                    -0.2254509 + 0.0j,
                ],
                [
                    -1.45753175e00 + 0.13348736j,
                    1.43589744e00 - 0.62176183j,
                    -3.15144231e00 + 4.10977564j,
                    5.33761070e-16 - 5.07957208j,
                    3.41973681e00 - 1.52820175j,
                    -2.04326923e00 + 0.0j,
                ],
                [
                    0.28469108 + 1.16619076j,
                    -1.31054131 - 0.27633859j,
                    -1.78561254 - 0.86823362j,
                    -0.17094017 - 0.57981758j,
                    -1.97500974 + 0.04361075j,
                    -1.45299145 + 0.0j,
                ],
                [
                    -0.84687993 - 0.66124746j,
                    1.18172589 - 1.80766318j,
                    -2.26497462 + 4.05126904j,
                    3.55583756 - 1.51664347j,
                    2.84198395 + 0.39907506j,
                    0.02284264 + 0.0j,
                ],
            ]
        )

        result = HCDF(5, 0.1)._calc_tivs(data[CHROMA_COLS].values)

        assert np.allclose(result, expected)

    def test_gaussian_blur(self, data: pd.DataFrame):
        expected = np.array(
            [
                [
                    -0.34657634 - 0.10074976j,
                    -0.34431436 - 0.10185624j,
                    -0.34030643 - 0.10382687j,
                    -0.33570068 - 0.10608419j,
                    -0.33173992 - 0.10802931j,
                    -0.32942092 - 0.10916909j,
                ],
                [
                    -0.34323816 - 0.10037259j,
                    -0.34095441 - 0.10148488j,
                    -0.33690876 - 0.10346592j,
                    -0.33225956 - 0.10573524j,
                    -0.32826128 - 0.10769076j,
                    -0.32592074 - 0.10883658j,
                ],
                [
                    -0.33729267 - 0.09969263j,
                    -0.33497031 - 0.10081547j,
                    -0.33085779 - 0.10281535j,
                    -0.32613156 - 0.10510642j,
                    -0.32206676 - 0.10708079j,
                    -0.31968804 - 0.10823756j,
                ],
                [
                    -0.32986083 - 0.09886351j,
                    -0.32749011 - 0.09999986j,
                    -0.32329381 - 0.10202386j,
                    -0.31847107 - 0.10434277j,
                    -0.31432295 - 0.10634126j,
                    -0.31189637 - 0.10751204j,
                ],
                [
                    -0.32243042 - 0.09802332j,
                    -0.32001118 - 0.09917288j,
                    -0.31573082 - 0.10122047j,
                    -0.31081124 - 0.10356663j,
                    -0.30657952 - 0.10558871j,
                    -0.30410493 - 0.1067732j,
                ],
                [
                    -0.31644409 - 0.09734781j,
                    -0.31398592 - 0.09850794j,
                    -0.30963809 - 0.10057442j,
                    -0.30464082 - 0.10294237j,
                    -0.30034199 - 0.10498333j,
                    -0.29782888 - 0.1061788j,
                ],
                [
                    -0.31314567 - 0.09697062j,
                    -0.31066615 - 0.09813668j,
                    -0.30628134 - 0.10021371j,
                    -0.30124146 - 0.10259387j,
                    -0.29690585 - 0.1046454j,
                    -0.29437162 - 0.105847j,
                ],
            ]
        )

        tivs = HCDF(5, 0.1)._calc_tivs(data[CHROMA_COLS].values)
        result = HCDF(5, 0.1)._gaussian_blur(tivs)

        assert np.allclose(result, expected)

    def test_compute_hcdf(self, data: pd.DataFrame):
        expected = np.array(
            [
                (0.0, 0.8671146244623705),
                (0.1, 0.023421976928086136),
                (0.2, 0.03374561190051528),
                (0.3, 0.037492097006850654),
                (0.4, 0.03384811297574506),
                (0.5, 0.023422556227886394),
                (0.6, 0.7846608976906223)
            ]
        )

        tivs = HCDF(5, 0.1)._calc_tivs(data[CHROMA_COLS].values)
        tivs = HCDF(5, 0.1)._gaussian_blur(tivs)
        result = HCDF(5, 0.1)._compute_hcdf(tivs)

        assert np.allclose(result, expected)

    def test_extract_peaks(self):
        """
        hcdf = np.array(
            [
                (0.0, 0.8671146244623705),
                (0.1, 0.023421976928086136),
                (0.2, 0.03374561190051528),
                (0.3, 0.037492097006850654),
                (0.4, 0.03384811297574506),
                (0.5, 0.023422556227886394),
                (0.6, 0.7846608976906223)
            ]
        )

        expected = np.array([
            (0.0, 0.8671146244623705),
            (0.3, 0.037492097006850654),
            (0.6, 0.7846608976906223)
        ])

        result = HCDF(5, 0.1)._extract_peaks(hcdf)

        assert np.allclose(result, expected)
        """
