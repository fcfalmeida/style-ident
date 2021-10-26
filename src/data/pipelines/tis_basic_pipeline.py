from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_BASIC_COLS
from src.features.tis_basic import TISBasic


class TISBasicPipeline(FeaturePipeline):
    def __init__(self) -> None:
        super().__init__([TISBasic()], TIS_BASIC_COLS)
