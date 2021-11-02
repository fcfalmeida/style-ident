from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_PIECE_FEATS
from src.features.tis_piece import TISPiece


class TISPieceSegmentedPipeline(FeaturePipeline):
    def __init__(self) -> None:
        super().__init__([TISPiece()], TIS_PIECE_FEATS)
