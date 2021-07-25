import pandas as pd
from src.features.feature_extractor import FeatureExtractor
from src.data.constants import CHROMA_COLS

class NormalizedChroma(FeatureExtractor):
    def l1_norm(self, chroma_vector: list[float]):
        norm = 0

        for q in chroma_vector:
            norm += abs(q)

        return norm

    def normalize_row(self, row: pd.Series):
        norm = self.l1_norm(row[CHROMA_COLS].values)

        if norm == 0:
            return row

        row[CHROMA_COLS] = row[CHROMA_COLS] / norm

        return row

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.apply(lambda row: self.normalize_row(row), axis=1)

        return result