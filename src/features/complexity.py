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

    def _chroma_std(self, chroma_vector: ArrayLike):
        std = np.std(chroma_vector, axis=1)

        rescale_factor = (1 / np.sqrt(12))

        return 1 - std / rescale_factor

    def _neg_slope(self, chroma_vector: ArrayLike):
        pitch_classes = np.array(list(range(12)))

        coef_matrix = np.vstack([pitch_classes, np.ones(len(pitch_classes))]).T

        # Get the slope values of the linear regression of each chroma vector
        slopes, _ = np.linalg.lstsq(coef_matrix, chroma_vector.T, rcond=None)[0]

        rescale_factor = -0.039

        return 1 - np.abs(slopes) / rescale_factor

    def _entropy(self, chroma_vector: ArrayLike):
        entropies = -1 / np.log2(12) * np.sum(chroma_vector * np.log2(chroma_vector, where=chroma_vector > 0), axis=1)

        return entropies

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().extract(data)