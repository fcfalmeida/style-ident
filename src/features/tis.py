import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import src.utils.math as math
from TIVlib import TIV, TIVCollection
from src.data.pipeline_task import PipelineTask
from src.data.constants import (
    CHROMA_COLS,
    COS_TONAL_DISP,
    EUC_TONAL_DISP,
    COS_DIST,
    EUC_DIST,
    DIMINISHED_QUALITTY,
    DISSONANCE,
    CHROMATICITY,
    DIATONICITY,
    DYADICITY,
    TRIADICITY,
    WHOLETONENESS,
    TIS_COLS,
    TIV_COL,
    HCDF_PEAK_IDX,
    HCDF_PEAK_INT
)


class TIS(PipelineTask):
    DIST_EUCLIDEAN = 0
    DIST_COSINE = 1

    def _tonal_dispersion(
        self, chroma_vectors: ArrayLike, tivs: TIVCollection,
            distance_type: int) -> pd.DataFrame:
        """Computes the tonal dispersion value of a `TIVCollection`.
        The tonal dispersion value is either the cosine or euclidean distance
        value between each `TIV` in the `TIVCollection` and the mean `TIV`.

        Args:
            chroma_vectors: Array of chroma vectors
            tivs: `TIVCollection` object which contains a list of TIVs
            distance_type: Type of distance measure to use. Can be one of
            DIST_COSINE or DIST_EUCLIDEAN
        Returns:
            An array of tonal dispersion values for each `TIV` in `tivs`
        """
        mean_chroma_vector = np.mean(chroma_vectors, axis=0)
        tonal_center = TIVCollection.from_pcp(mean_chroma_vector)

        tonal_disp = None

        if distance_type == self.DIST_EUCLIDEAN:
            tonal_disp = np.linalg.norm(
                tivs.vectors - tonal_center.vectors, axis=1
            )
        elif distance_type == self.DIST_COSINE:
            tonal_disp = math.complex_cosine_dist(
                tivs.vectors, tonal_center.vectors
            )

        return tonal_disp

    def _distance(
            self, tivs: TIVCollection, distance_type: int) -> pd.DataFrame:
        dist = None

        if distance_type == self.DIST_EUCLIDEAN:
            dist = np.linalg.norm(
                tivs.vectors[0:-1, ] - tivs.vectors[1:, ], axis=1
            )
        elif distance_type == self.DIST_COSINE:
            dist = math.complex_cosine_dist(
                tivs.vectors[0:-1, ], tivs.vectors[1:, ]
            )

        dist = np.append(dist, 0)

        return dist

    def _tiv_mags(self, tivs: TIVCollection):
        return np.abs(tivs.vectors)

    def _descriptor(self, tiv_mags: ArrayLike, k: int) -> ArrayLike:
        return tiv_mags[:, k] / TIV.weights[k]

    def _dissonance(self, tivs: TIVCollection):
        return 1 - (np.linalg.norm(tivs.vectors, axis=1)/np.sqrt(np.sum(
            np.dot(TIV.weights, TIV.weights)))
        )

    def _hcdf_peak_interval(self, hcdf_peak_indexes: ArrayLike):
        intervals = hcdf_peak_indexes[1:] - hcdf_peak_indexes[0:-1]

        intervals = np.append(intervals, 0)

        return intervals

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data_cpy = data.copy()

        tivs = TIVCollection.from_pcp(data_cpy[CHROMA_COLS].values.T)
        mags = self._tiv_mags(tivs)

        data_cpy[TIV_COL] = tivs.tivlist

        grouped = data_cpy.groupby('piece')

        for piece, group in grouped:
            group_tivs = TIVCollection(group[TIV_COL].values)

            cos_tonal_disp = self._tonal_dispersion(
                group[CHROMA_COLS].values,
                group_tivs,
                self.DIST_COSINE
            )

            euc_tonal_disp = self._tonal_dispersion(
                group[CHROMA_COLS].values,
                group_tivs,
                self.DIST_EUCLIDEAN
            )

            cos_dist = self._distance(group_tivs, self.DIST_COSINE)
            euc_dist = self._distance(group_tivs, self.DIST_EUCLIDEAN)

            data_cpy.loc[piece, COS_TONAL_DISP] = cos_tonal_disp
            data_cpy.loc[piece, EUC_TONAL_DISP] = euc_tonal_disp
            data_cpy.loc[piece, COS_DIST] = cos_dist
            data_cpy.loc[piece, EUC_DIST] = euc_dist

            data_cpy.loc[piece, HCDF_PEAK_INT] = self._hcdf_peak_interval(
                group[HCDF_PEAK_IDX].values
            )

        data_cpy[DISSONANCE] = self._dissonance(tivs)
        data_cpy[CHROMATICITY] = self._descriptor(mags, 0)
        data_cpy[DYADICITY] = self._descriptor(mags, 1)
        data_cpy[TRIADICITY] = self._descriptor(mags, 2)
        data_cpy[DIMINISHED_QUALITTY] = self._descriptor(mags, 3)
        data_cpy[DIATONICITY] = self._descriptor(mags, 4)
        data_cpy[WHOLETONENESS] = self._descriptor(mags, 5)

        return data_cpy[TIS_COLS]
