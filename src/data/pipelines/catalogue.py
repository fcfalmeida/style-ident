from src.data.pipelines.tis_complexity_segmented_pipeline import (
    TISComplexitySegmentedPipeline,
)
from src.data.pipelines.tis_basic_segmented_pipeline import (
    TISBasicSegmentedPipeline,
)
from src.data.pipelines.tis_complexity_res_pipeline import (
    TISComplexityResPipeline,
)
from src.data.pipelines.tis_basic_res_pipeline import TISBasicResPipeline
from src.data.pipelines.harm_rhythm_pipeline import HarmRhythmPipeline
from src.data.pipelines.template_based_pipeline import TemplateBasedPipeline
from src.data.pipelines.complexity_pipeline import ComplexityPipeline


segmented_pipelines = {
    'tis_complexity_segmented': TISComplexitySegmentedPipeline,
    'tis_basic_segmented': TISBasicSegmentedPipeline,
    'harm_rhythm': HarmRhythmPipeline,
}

res_pipelines = {
    'tis_complexity_res': TISComplexityResPipeline,
    'tis_basic_res': TISBasicResPipeline,
    'complexity': ComplexityPipeline,
    'template_based': TemplateBasedPipeline,
}
