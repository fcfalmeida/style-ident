from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_COMPLEXITY
from src.features.chroma_resolution import ChromaResolution
from src.features.tis_vertical import TISVertical
from src.features.tis_horizontal import TISHorizontal


class TISComplexityResPipeline(FeaturePipeline):
    def __init__(self, chroma_resolution: float) -> None:
        super().__init__(
            [TISVertical(), TISHorizontal()],
            TIS_COMPLEXITY,
            [ChromaResolution(chroma_resolution)]
        )
