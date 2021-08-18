import pandas as pd
from src.data.pipeline_task import PipelineTask

class PipelineTaskGroup(PipelineTask):
    def __init__(self) -> None:
        self.tasks = []

    def add_task(self, t: PipelineTask) -> None:
        self.tasks.append(t)

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()

        for t in self.tasks:
            df = t.run(data_copy)
            data_copy = data_copy.join(df)

        return data_copy