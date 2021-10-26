from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TEMPLATE_FEATS
from src.features.chroma_resolution import ChromaResolution
from src.features.normalized_chroma import NormalizedChroma
from src.features.template_based import TemplateBased


class TemplateBasedPipeline(FeaturePipeline):
    def __init__(self, chroma_resolution: float) -> None:
        super().__init__(
            [TemplateBased()],
            TEMPLATE_FEATS,
            [ChromaResolution(chroma_resolution), NormalizedChroma()]
        )
