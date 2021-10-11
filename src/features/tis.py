import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from TIVlib import TIV, TIVCollection
from src.data.pipeline_task import PipelineTask
from src.data.constants import CHROMA_COLS, DIMINISHED_QUALITTY, DISSONANCE, CHROMATICITY, \
    DIATONICITY, DYADICITY, TRIADICITY, WHOLETONENESS


class TIS(PipelineTask):
    DIST_EUCLIDEAN = 0
    DIST_COSINE = 1

    def _tonal_dispersion(
        self, chroma_vectors: pd.DataFrame, distance_type: int
    ) -> pd.DataFrame:
        mean_chroma_vector = chroma_vectors.mean()

        tonal_disp = chroma_vectors.copy()

    def _tiv_mags(self, tivs: TIVCollection):
        return np.abs(tivs.vectors)

    def _descriptor(self, tiv_mags: ArrayLike, k: int) -> ArrayLike:
        return tiv_mags[:, k] / TIV.weights[k]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()

        tivs = TIVCollection.from_pcp(data_copy[CHROMA_COLS].values.T)
        mags = self._tiv_mags(tivs)
        """
        grouped = data_copy.groupby(['piece', 'time'])[CHROMA_COLS].agg([
            self._tonal_dispersion
        ])
        """

        # data_copy[DISSONANCE] = tivs.dissonance()
        data_copy[CHROMATICITY] = self._descriptor(mags, 0)
        data_copy[DYADICITY] = self._descriptor(mags, 1)
        data_copy[TRIADICITY] = self._descriptor(mags, 2)
        data_copy[DIMINISHED_QUALITTY] = self._descriptor(mags, 3)
        data_copy[DIATONICITY] = self._descriptor(mags, 4)
        data_copy[WHOLETONENESS] = self._descriptor(mags, 5)

        return data_copy
