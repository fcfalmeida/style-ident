from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import TIS_BASIC_COLS, TIS_PIECE_FEATS
from src.features.tis_basic import TISBasic
from src.features.tis_piece import TISPiece


class TISCombinedSegmentedPipeline(FeaturePipeline):
    def __init__(self) -> None:
        super().__init__([TISBasic(), TISPiece()], TIS_BASIC_COLS + TIS_PIECE_FEATS)
