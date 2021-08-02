import pandas as pd
from numpy.typing import ArrayLike
from src.features.feature_extractor import FeatureExtractor
from src.data.constants import CHROMA_COLS, INTERVAL_COLS

class Intervals(FeatureExtractor):
    TEMPLATES = {
        INTERVAL_COLS[0]: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        INTERVAL_COLS[1]: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        INTERVAL_COLS[2]: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        INTERVAL_COLS[3]: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        INTERVAL_COLS[4]: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        INTERVAL_COLS[5]: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    }

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()

        for interval_cat in self.TEMPLATES.keys():
            data_copy[interval_cat] = self.calc_interval_feat(data[CHROMA_COLS].values, interval_cat)

        return data_copy[INTERVAL_COLS]

    def calc_interval_feat(self, chroma_vector: ArrayLike, interval_cat: str) -> ArrayLike:
        template = self.TEMPLATES[interval_cat]

        likelihood_sum = 0
        for q in range(12):
            likelihood = 1

            for k in range(12):
                chroma_value = chroma_vector[:, (q + k) % 12]

                if template[k]:
                    likelihood *= chroma_value

            likelihood_sum += likelihood

        return likelihood_sum