from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_BASIC_FEATS
from src.features.chroma_resolution import ChromaResolution
from src.features.tis import TIS


class TISBasicResPipeline(FeaturePipeline):
    def __init__(self, chroma_resolution: float) -> None:
        super().__init__(
            [TIS()],
            TIS_BASIC_FEATS,
            [ChromaResolution(chroma_resolution)]
        )
