import pandas as pd
from src.features.feature_extractor import FeatureExtractor
from src.data.constants import CHROMA_COLS

class RemoveChroma(FeatureExtractor):
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=CHROMA_COLS)