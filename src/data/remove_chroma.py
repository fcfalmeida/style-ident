import pandas as pd
from src.data.pipeline_task import PipelineTask
from src.data.constants import CHROMA_COLS


class RemoveChroma(PipelineTask):
    """When run, this task removes all columns pertaining to chroma
    feature values from the specified `DataFrame`object
    """

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=CHROMA_COLS)
