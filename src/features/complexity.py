import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats.mstats import gmean
from src.features.feature_extractor import FeatureExtractor
import src.utils.math as math

class Complexity(FeatureExtractor):
    def _sort_chroma_fifths(self, chroma_vector: ArrayLike):
        sorted = chroma_vector.copy()

        for q in range(12):
            sorted[:, q] = chroma_vector[:, (q * 7) % 12]

        return sorted

    def _sum_chroma_diff(self, chroma_vector: ArrayLike):
        zero_chromas = np.where(~chroma_vector.any(axis=1))[0]
        diff_sum = np.zeros(chroma_vector.shape[0])

        for q in range(12):
            diff = chroma_vector[:, (q + 1) % 12] - chroma_vector[:, q]
            diff = np.abs(diff)

            diff_sum += diff

        diff_sum = 1 - diff_sum / 2
        diff_sum.put(zero_chromas, 0)

        return diff_sum

    def _chroma_std(self, chroma_vector: ArrayLike):
        zero_chromas = np.where(~chroma_vector.any(axis=1))[0]
        std = np.std(chroma_vector, axis=1)

        rescale_factor = 1 / np.sqrt(12)

        std = 1 - std / rescale_factor
        std.put(zero_chromas, 0)

        return std

    def _neg_slope(self, chroma_vector: ArrayLike):
        zero_chromas = np.where(~chroma_vector.any(axis=1))[0]
        chroma_vector = np.sort(-chroma_vector, axis=1)
        pitch_classes = np.array(list(range(12)))

        coef_matrix = np.vstack([pitch_classes, np.ones(len(pitch_classes))]).T

        # Get the slope values of the linear regression of each chroma vector
        slopes, _ = np.linalg.lstsq(coef_matrix, chroma_vector.T, rcond=None)[0]

        rescale_factor = 0.039

        slopes = 1 - np.abs(slopes) / rescale_factor
        slopes.put(zero_chromas, 0)

        return slopes

    def _entropy(self, chroma_vector: ArrayLike):
        entropies = -1 / np.log2(12) * np.sum(chroma_vector * np.log2(chroma_vector, where=chroma_vector > 0), axis=1)

        return entropies

    def _non_sparseness(self, chroma_vector: ArrayLike):
        zero_chromas = np.where(~chroma_vector.any(axis=1))[0]
        l1 = math.l1_norm(chroma_vector)
        l2 = math.l2_norm(chroma_vector)

        non_sparse = 1 - (np.sqrt(12) - np.divide(l1, l2, where=l2 > 0)) / (np.sqrt(12) - 1)
        non_sparse.put(zero_chromas, 0)

        return non_sparse

    def _flatness(self, chroma_vector: ArrayLike):
        with np.errstate(divide='ignore'):
            geom_mean = gmean(chroma_vector, axis=1)
        
        arith_mean = np.mean(chroma_vector, axis=1)

        return np.divide(geom_mean, arith_mean, where=arith_mean > 0)

    def _angular_deviation(self, chroma_vector: ArrayLike):
        zero_chromas = np.where(~chroma_vector.any(axis=1))[0]
        fifth_sorted = self._sort_chroma_fifths(chroma_vector)
        pitch_class_dist = 0

        for q in range(12):
            pitch_class_dist += fifth_sorted[:, q] * np.exp(2j * np.pi * q / 12)

        pitch_class_dist = np.abs(pitch_class_dist)
        pitch_class_dist = np.sqrt(1 - pitch_class_dist)
        pitch_class_dist.put(zero_chromas, 0)

        return pitch_class_dist

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().extract(data)