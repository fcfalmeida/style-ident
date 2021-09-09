import pandas as pd
import numpy as np
from src.data.pipeline_task import PipelineTask

class MeanAndStd(PipelineTask):
    """This task computes the mean and standard deviation of all columns in a `DataFrame` object.
    """
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        def std(data):
            return np.std(data)

        temp = data.groupby('piece').agg([np.mean, std])
        temp.columns = temp.columns.to_flat_index()
        
        return temp
