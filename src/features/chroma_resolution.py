import pandas as pd
from src.data.pipeline_task import PipelineTask
from src.utils.converters import sec_to_ms

class ChromaResolution(PipelineTask):
    GLOBAL = 0

    def __init__(self, resolution: float) -> None:
        super().__init__()

        self.resolution = resolution

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data['time'] = pd.to_timedelta(data['time'], unit='s')

        if self.resolution == ChromaResolution.GLOBAL:
            return data.groupby('piece').sum()

        rule = f'{sec_to_ms(self.resolution)}ms'

        return data.groupby('piece').resample(rule, on='time').sum()
