import pandas as pd
from abc import ABC, abstractmethod

class PipelineTask(ABC):
    @abstractmethod
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        pass #pragma: no cover