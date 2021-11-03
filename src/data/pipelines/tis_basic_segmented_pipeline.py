from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_BASIC_FEATS
from src.features.tis_vertical import TISVertical


class TISBasicSegmentedPipeline(FeaturePipeline):
    def __init__(self) -> None:
        super().__init__([TISVertical()], TIS_BASIC_FEATS)
