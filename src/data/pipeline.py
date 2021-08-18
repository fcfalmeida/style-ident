import pandas as pd
from src.data.pipeline_task import PipelineTask

class Pipeline():
    def __init__(self) -> None:
        self.steps = []

    def add_task(self, step: PipelineTask) -> None:
        self.steps.append(step)

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        for s in self.steps:
            df = s.run(df)

        return df