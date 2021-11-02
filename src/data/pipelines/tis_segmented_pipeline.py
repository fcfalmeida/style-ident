from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_COLS
from src.features.tis import TIS


class TISSegmentedPipeline(FeaturePipeline):
    def __init__(self) -> None:
        super().__init__([TIS()], TIS_COLS)
