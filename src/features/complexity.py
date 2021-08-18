import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from src.data.pipeline_task import PipelineTask
from scipy.stats.mstats import gmean
from src.data.constants import CHROMA_COLS, COMPLEXITY_DIFF, COMPLEXITY_STD, \
    COMPLEXITY_SLOPE, COMPLEXITY_ENTROPY, COMPLEXITY_NON_SPARSENESS, \
    COMPLEXITY_FLATNESS, COMPLEXITY_FIFTH_ANG_DEV, COMPLEXITY_COLS
import src.utils.math as math

class Complexity(PipelineTask):
    def _null_chroma_returns_zero(feature):
        def wrapper(self, chroma_vector):
            null_chromas = np.where(~chroma_vector.any(axis=1))[0]
            result = feature(self, chroma_vector)
            result.put(null_chromas, 0)

            return result

        return wrapper

    def _sort_chroma_fifths(self, chroma_vector: ArrayLike):
        sorted = chroma_vector.copy()

        for q in range(12):
            sorted[:, q] = chroma_vector[:, (q * 7) % 12]

        return sorted

    @_null_chroma_returns_zero
    def _sum_chroma_diff(self, chroma_vector: ArrayLike):
        diff_sum = np.zeros(chroma_vector.shape[0])

        for q in range(12):
            diff = chroma_vector[:, (q + 1) % 12] - chroma_vector[:, q]
            diff = np.abs(diff)

            diff_sum += diff

        diff_sum = 1 - diff_sum / 2

        return diff_sum

    @_null_chroma_returns_zero
    def _chroma_std(self, chroma_vector: ArrayLike):
        std = np.std(chroma_vector, axis=1)

        rescale_factor = 1 / np.sqrt(12)

        std = 1 - std / rescale_factor

        return std

    @_null_chroma_returns_zero
    def _neg_slope(self, chroma_vector: ArrayLike):
        chroma_vector = np.sort(-chroma_vector, axis=1)
        pitch_classes = np.array(list(range(12)))

        coef_matrix = np.vstack([pitch_classes, np.ones(len(pitch_classes))]).T

        # Get the slope values of the linear regression of each chroma vector
        slopes, _ = np.linalg.lstsq(coef_matrix, chroma_vector.T, rcond=None)[0]

        rescale_factor = 0.039

        slopes = 1 - np.abs(slopes) / rescale_factor

        return slopes

    def _entropy(self, chroma_vector: ArrayLike):
        entropies = -1 / np.log2(12) * np.sum(chroma_vector * np.log2(chroma_vector, where=chroma_vector > 0), axis=1)

        return entropies

    @_null_chroma_returns_zero
    def _non_sparseness(self, chroma_vector: ArrayLike):
        l1 = math.l1_norm(chroma_vector)
        l2 = math.l2_norm(chroma_vector)

        non_sparse = 1 - (np.sqrt(12) - np.divide(l1, l2, where=l2 > 0)) / (np.sqrt(12) - 1)

        return non_sparse

    @_null_chroma_returns_zero
    def _flatness(self, chroma_vector: ArrayLike):
        with np.errstate(divide='ignore'):
            geom_mean = gmean(chroma_vector, axis=1)
            
        arith_mean = np.mean(chroma_vector, axis=1)

        ratio = np.divide(geom_mean, arith_mean, where=arith_mean > 0)

        return ratio

    @_null_chroma_returns_zero
    def _angular_deviation(self, chroma_vector: ArrayLike):
        fifth_sorted = self._sort_chroma_fifths(chroma_vector)
        pitch_class_dist = 0

        for q in range(12):
            pitch_class_dist += fifth_sorted[:, q] * np.exp(2j * np.pi * q / 12)

        pitch_class_dist = np.abs(pitch_class_dist)
        pitch_class_dist = np.sqrt(1 - pitch_class_dist)

        return pitch_class_dist

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data_cpy = data.copy()

        data_cpy[COMPLEXITY_DIFF] = self._sum_chroma_diff(data[CHROMA_COLS].values)
        data_cpy[COMPLEXITY_STD] = self._chroma_std(data[CHROMA_COLS].values)
        data_cpy[COMPLEXITY_SLOPE] = self._neg_slope(data[CHROMA_COLS].values)
        data_cpy[COMPLEXITY_ENTROPY] = self._entropy(data[CHROMA_COLS].values)
        data_cpy[COMPLEXITY_NON_SPARSENESS] = self._non_sparseness(data[CHROMA_COLS].values)
        data_cpy[COMPLEXITY_FLATNESS] = self._flatness(data[CHROMA_COLS].values)
        data_cpy[COMPLEXITY_FIFTH_ANG_DEV] = self._angular_deviation(data[CHROMA_COLS].values)

        return data_cpy[COMPLEXITY_COLS]