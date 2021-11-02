from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_BASIC_COLS, TIS_PIECE_FEATS
from src.features.chroma_resolution import ChromaResolution
from src.features.tis_basic import TISBasic
from src.features.tis_piece import TISPiece


class TISCombinedResPipeline(FeaturePipeline):
    def __init__(self, chroma_resolution: float) -> None:
        super().__init__(
            [TISBasic(), TISPiece()],
            TIS_BASIC_COLS + TIS_PIECE_FEATS,
            [ChromaResolution(chroma_resolution)]
        )
