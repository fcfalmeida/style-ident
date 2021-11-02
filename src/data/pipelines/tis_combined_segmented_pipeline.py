from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_BASIC_COLS, TIS_COLS
from src.features.tis_basic import TISBasic
from src.features.tis import TIS


class TISCombinedSegmentedPipeline(FeaturePipeline):
    def __init__(self) -> None:
        super().__init__([TISBasic(), TIS()], TIS_BASIC_COLS + TIS_COLS)
