import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from src.features.feature_extractor import FeatureExtractor

class Complexity(FeatureExtractor):
    def _sort_chroma_fifths(self, chroma_vector: ArrayLike):
        sorted = chroma_vector.copy()

        for q in range(12):
            sorted[:, q] = chroma_vector[:, (q * 7) % 12]

        return sorted

    def _sum_chroma_diff(self, chroma_vector: ArrayLike):
        diff_sum = 0

        for q in range(12):
            diff = chroma_vector[:, (q + 1) % 12] - chroma_vector[:, q]
            diff = np.abs(diff)

            diff_sum += diff

        return 1 - diff_sum / 2


    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().extract(data)