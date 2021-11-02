from src.data.pipelines.tis_piece_segmented_pipeline import (
    TISPieceSegmentedPipeline,
)
from src.data.pipelines.tis_basic_segmented_pipeline import (
    TISBasicSegmentedPipeline,
)
from src.data.pipelines.tis_combined_segmented_pipeline import (
    TISCombinedSegmentedPipeline,
)
from src.data.pipelines.tis_piece_res_pipeline import TISPieceResPipeline
from src.data.pipelines.tis_basic_res_pipeline import TISBasicResPipeline
from src.data.pipelines.tis_combined_res_pipeline import TISCombinedResPipeline
from src.data.pipelines.harm_rhythm_pipeline import HarmRhythmPipeline
from src.data.pipelines.template_based_pipeline import TemplateBasedPipeline
from src.data.pipelines.template_complexity import TemplateComplexityPipeline
from src.data.pipelines.complexity_pipeline import ComplexityPipeline

pipeline_catalogue = {
    'tis_piece_segmented': TISPieceSegmentedPipeline,
    'tis_basic_segmented': TISBasicSegmentedPipeline,
    'tis_combined_segmented': TISCombinedSegmentedPipeline,
    'tis_piece_res': TISPieceResPipeline,
    'tis_basic_res': TISBasicResPipeline,
    'tis_combined_res': TISCombinedResPipeline,
    'harm_rhythm': HarmRhythmPipeline,
    'complexity': ComplexityPipeline,
    'template_based': TemplateBasedPipeline,
    'template_complexity': TemplateComplexityPipeline,
}
