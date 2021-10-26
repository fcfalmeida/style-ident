from src.data.pipelines.tis_pipeline import TISPipeline
from src.data.pipelines.tis_basic_pipeline import TISBasicPipeline
from src.data.pipelines.tis_combined_pipeline import TISCombinedPipeline
from src.data.pipelines.template_based_pipeline import TemplateBasedPipeline

pipeline_catalogue = {
    'tis': TISPipeline(),
    'tis_basic': TISBasicPipeline(),
    'tis_combined': TISCombinedPipeline(),
    'template_based': TemplateBasedPipeline
}
