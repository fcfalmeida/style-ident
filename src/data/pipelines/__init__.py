from src.data.pipelines.tis_pipeline import TISPipeline
from src.data.pipelines.tis_basic_pipeline import TISBasicPipeline
from src.data.pipelines.tis_combined_pipeline import TISCombinedPipeline

pipeline_catalogue = {
    'tis': TISPipeline(),
    'tis_basic': TISBasicPipeline(),
    'tis_combined': TISCombinedPipeline()
}
