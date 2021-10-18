from numpy.typing import ArrayLike
import pandas as pd
import numpy as np
import copy
from lib.HCDF.HCDF import harmonic_change
from src.data.constants import CHROMA_COLS, HCDF_PEAK_IDX, HCDF_PEAK_MAG
from src.data.pipeline_task import PipelineTask


class HCDFSegmentation(PipelineTask):
    def _get_window_right_bounds(
        self, changes: ArrayLike, n_chromas: ArrayLike
    ):
        right_bounds = copy.deepcopy(changes)

        # Remove 0 and subtract 1 to all elements
        right_bounds = right_bounds[1:] - 1
        right_bounds = np.append(right_bounds, n_chromas - 1)

        return right_bounds

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        # Grouped by piece with a column containing the tuple
        # (changes, hcdf_changes, harmonic_function)
        grouped = data.groupby('piece', sort=False)

        segmented = []

        for piece, group in grouped:
            # Get left bound of windows
            peak_indexes, peak_mags, _ = harmonic_change(
                group[CHROMA_COLS].values
            )

            right_bounds = self._get_window_right_bounds(
                peak_indexes, group.shape[0]
            )

            for i in range(peak_indexes.size):
                left_bound_idx = peak_indexes[i]
                right_bound_idx = right_bounds[i]

                piece, time = group.iloc[left_bound_idx].name
                row = group[left_bound_idx:right_bound_idx].sum().to_dict()
                row['piece'] = piece
                row['time'] = time
                row[HCDF_PEAK_IDX] = peak_indexes[i]
                row[HCDF_PEAK_MAG] = peak_mags[i]

                segmented.append(row)

        segmented_df = pd.DataFrame(segmented)
        segmented_df = segmented_df.set_index(['piece', 'time'])

        return segmented_df
