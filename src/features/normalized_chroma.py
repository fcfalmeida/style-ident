import pandas as pd
from src.data.pipeline_task import PipelineTask

class NormalizedChroma(PipelineTask):
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalizes chroma vectors by diving each vector by its manhattan norm

        Args:
            data (pd.DataFrame): dataframe containing chroma vectors

        Returns:
            pd.DataFrame: dataframe with normalized chroma vectors
        """
        return data.div(data.abs().sum(axis=1), axis=0).fillna(0)