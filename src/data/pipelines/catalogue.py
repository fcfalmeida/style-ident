from src.data.pipelines.tis_segmented_pipeline import TISSegmentedPipeline
from src.data.pipelines.tis_basic_segmented_pipeline import (
    TISBasicSegmentedPipeline,
)
from src.data.pipelines.tis_combined_segmented_pipeline import (
    TISCombinedSegmentedPipeline,
)
from src.data.pipelines.tis_res_pipeline import TISResPipeline
from src.data.pipelines.tis_basic_res_pipeline import TISBasicResPipeline
from src.data.pipelines.tis_combined_res_pipeline import TISCombinedResPipeline
from src.data.pipelines.harm_rhythm_pipeline import HarmRhythmPipeline
from src.data.pipelines.template_based_pipeline import TemplateBasedPipeline
from src.data.pipelines.template_complexity import TemplateComplexityPipeline
from src.data.pipelines.complexity_pipeline import ComplexityPipeline

pipeline_catalogue = {
    'tis_segmented': TISSegmentedPipeline,
    'tis_basic_segmented': TISBasicSegmentedPipeline,
    'tis_combined_segmented': TISCombinedSegmentedPipeline,
    'tis_res': TISResPipeline,
    'tis_basic_res': TISBasicResPipeline,
    'tis_combined_res': TISCombinedResPipeline,
    'harm_rhythm': HarmRhythmPipeline,
    'complexity': ComplexityPipeline,
    'template_based': TemplateBasedPipeline,
    'template_complexity': TemplateComplexityPipeline,
}
