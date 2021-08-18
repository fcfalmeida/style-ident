import pandas as pd
from src.features.feature_extractor import FeatureExtractor

class FeatureGroup(FeatureExtractor):
    def __init__(self) -> None:
        self.extractors = []

    def add_extractor(self, extractor: FeatureExtractor) -> None:
        self.extractors.append(extractor)

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()

        for extractor in self.extractors:
            df = extractor.extract(data_copy)
            data_copy = data_copy.join(df)

        return data_copy
