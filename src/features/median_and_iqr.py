import pandas as pd
import numpy as np
from scipy.stats import iqr
from src.data.pipeline_task import PipelineTask


class MedianAndIQR(PipelineTask):
    """This task computes the median and IQR of all columns in a
    `DataFrame` object."""

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        def std(data):
            return np.std(data)

        temp = data.groupby("piece").agg([np.median, iqr])
        temp.columns = temp.columns.to_flat_index()

        return temp
