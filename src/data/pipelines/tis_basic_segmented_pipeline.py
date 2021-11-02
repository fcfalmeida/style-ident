from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_BASIC_FEATS
from src.features.tis import TIS


class TISBasicSegmentedPipeline(FeaturePipeline):
    def __init__(self) -> None:
        super().__init__([TIS()], TIS_BASIC_FEATS)
