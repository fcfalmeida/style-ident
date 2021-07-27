import pandas as pd
from src.features.feature_extractor import FeatureExtractor

class Pipeline:
    def __init__(self) -> None:
        self.steps = []

    def add_step(self, extractor: FeatureExtractor) -> None:
        self.steps.append(extractor)

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        for s in self.steps:
            df = s.extract(df)

        return df