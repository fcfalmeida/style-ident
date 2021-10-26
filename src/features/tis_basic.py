import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from TIVlib import TIV, TIVCollection
from src.data.tasks.pipeline_task import PipelineTask
from src.data.constants import (
    CHROMA_COLS,
    DIMINISHED_QUALITTY,
    DISSONANCE,
    CHROMATICITY,
    DIATONICITY,
    DYADICITY,
    TRIADICITY,
    WHOLETONENESS,
    TIS_BASIC_COLS
)


class TISBasic(PipelineTask):
    def _tiv_mags(self, tivs: TIVCollection):
        return np.abs(tivs.vectors)

    def _descriptor(self, tiv_mags: ArrayLike, k: int) -> ArrayLike:
        return tiv_mags[:, k] / TIV.weights[k]

    def _dissonance(self, tivs: TIVCollection):
        return 1 - (np.linalg.norm(tivs.vectors, axis=1)/np.sqrt(np.sum(
            np.dot(TIV.weights, TIV.weights)))
        )

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data_cpy = data.copy()

        tivs = TIVCollection.from_pcp(data_cpy[CHROMA_COLS].values.T)
        mags = self._tiv_mags(tivs)

        data_cpy[DISSONANCE] = self._dissonance(tivs)
        data_cpy[CHROMATICITY] = self._descriptor(mags, 0)
        data_cpy[DYADICITY] = self._descriptor(mags, 1)
        data_cpy[TRIADICITY] = self._descriptor(mags, 2)
        data_cpy[DIMINISHED_QUALITTY] = self._descriptor(mags, 3)
        data_cpy[DIATONICITY] = self._descriptor(mags, 4)
        data_cpy[WHOLETONENESS] = self._descriptor(mags, 5)

        return data_cpy[TIS_BASIC_COLS]
