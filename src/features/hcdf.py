import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage.filters import gaussian_filter
from TIVlib import TIV
from src.data.pipeline_task import PipelineTask
from src.data.constants import CHROMA_COLS


class HCDF(PipelineTask):
    def __init__(
        self,
        gaussian_sigma: float = 5,
        chroma_resolution: float = 0.1
    ) -> None:
        self.gaussian_sigma = gaussian_sigma
        self.chroma_resolution = chroma_resolution

    def _calc_tivs(self, chroma_vector: ArrayLike) -> ArrayLike:
        # Supress 0 division warning in TIV calculation
        with np.errstate(divide="ignore", invalid="ignore"):
            tivs = np.apply_along_axis(
                lambda c: TIV.from_pcp(c).vector, 1, chroma_vector
            )
            tivs = np.nan_to_num(tivs)

            return tivs

    def _gaussian_blur(self, tivs: ArrayLike) -> ArrayLike:
        blurred_tivs = gaussian_filter(
            np.transpose(tivs), sigma=self.gaussian_sigma
        )

        return np.transpose(blurred_tivs)

    def _calc_distance(
        self, tivs1: ArrayLike, tivs2: ArrayLike = None
    ) -> ArrayLike:
        if tivs2 is None:
            tivs2 = np.zeros_like(tivs1)

        try:
            return np.linalg.norm(tivs1 - tivs2, axis=1)
        except np.AxisError:
            return np.linalg.norm(tivs1 - tivs2)

    def _compute_hcdf(self, tivs: ArrayLike) -> ArrayLike:
        hcdf = self._calc_distance(tivs[2:, :], tivs[:-2, :])

        # Append HCDF values for the first and last TIVs
        hcdf = np.insert(hcdf, 0, self._calc_distance(tivs[0]))
        hcdf = np.append(hcdf, self._calc_distance(tivs[-1]))

        # Map HCDF function as a (x, y) tuple where x represents time
        # and y the value of the HCDF function at that time
        hcdf = np.array([
            (self.chroma_resolution * i, value) for i, value in enumerate(hcdf)
        ])

        return hcdf

    def _extract_peaks(self, hcdf: ArrayLike) -> ArrayLike:
        pass

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        tivs = self._calc_tivs(data[CHROMA_COLS].values)
        smoothed_tivs = self._gaussian_blur(tivs)
        hcdf = self._compute_hcdf(smoothed_tivs)

        return hcdf
