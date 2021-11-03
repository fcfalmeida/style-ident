from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_COMPLEXITY
from src.features.tis_vertical import TISVertical
from src.features.tis_horizontal import TISHorizontal


class TISComplexitySegmentedPipeline(FeaturePipeline):
    def __init__(self) -> None:
        super().__init__([TISVertical(), TISHorizontal()], TIS_COMPLEXITY)
