import pandas as pd
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        pass