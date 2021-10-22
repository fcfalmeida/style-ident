from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants import TIS_COLS
from src.features.tis import TIS


class TISPipeline(FeaturePipeline):
    def __init__(self) -> None:
        super().__init__([TIS()], TIS_COLS)
