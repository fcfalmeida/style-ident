from src.data.pipelines.pipeline import Pipeline
from src.data.pipelines.pipeline_task import PipelineTask
from src.data.pipeline_task_group import PipelineTaskGroup
from src.data.select_columns import SelectColumns
from src.features.mean_and_std import MeanAndStd


class FeaturePipeline(Pipeline):
    def __init__(
        self,
        feature_tasks: list[PipelineTask],
        feature_cols: list[str],
            prep_tasks: list[PipelineTask] = []) -> None:
        super().__init__()

        for task in prep_tasks:
            self.add_task(task)

        group = PipelineTaskGroup()

        for task in feature_tasks:
            group.add_task(task)

        self.add_task(group)

        self.add_task(SelectColumns(feature_cols))
        self.add_task(MeanAndStd())
