from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_FEATS, TIS_PIECE_FEATS
from src.features.tis import TIS
from src.features.tis_piece import TISPiece


class TISCombinedSegmentedPipeline(FeaturePipeline):
    def __init__(self) -> None:
        super().__init__([TIS(), TISPiece()], TIS_FEATS + TIS_PIECE_FEATS)
