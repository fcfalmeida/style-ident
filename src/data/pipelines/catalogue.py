from src.data.pipelines.tis_pipeline import TISPipeline
from src.data.pipelines.tis_basic_pipeline import TISBasicPipeline
from src.data.pipelines.tis_combined_pipeline import TISCombinedPipeline
from src.data.pipelines.template_based_pipeline import TemplateBasedPipeline
from src.data.pipelines.template_complexity import TemplateComplexityPipeline
from src.data.pipelines.complexity_pipeline import ComplexityPipeline

pipeline_catalogue = {
    'tis': TISPipeline(),
    'tis_basic': TISBasicPipeline(),
    'tis_combined': TISCombinedPipeline(),
    'complexity': ComplexityPipeline,
    'template_based': TemplateBasedPipeline,
    'template_complexity': TemplateComplexityPipeline
}
