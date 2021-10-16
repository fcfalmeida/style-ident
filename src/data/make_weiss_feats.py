import click
import pathlib
import pandas as pd
from src.data.pipeline import Pipeline
from src.data.constants import CHROMA_COLS
from src.data.remove_columns import RemoveColumns
from src.features.chroma_resolution import ChromaResolution
from src.features.normalized_chroma import NormalizedChroma
from src.features.template_based import TemplateBased
from src.features.complexity import Complexity
from src.data.pipeline_task_group import PipelineTaskGroup
from src.features.mean_and_std import MeanAndStd
from src.utils.formatters import format_chroma_resolution


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    RESOLUTIONS = [0.1, 0.5, 10, ChromaResolution.GLOBAL]
    COL_NAMES = ["piece", "time"]
    COL_NAMES.extend(CHROMA_COLS)

    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            data = pd.read_csv(path, dtype={"piece": str})
            data = data.fillna(method="ffill")

            for resolution in RESOLUTIONS:
                pipeline = make_pipeline(resolution)

                processed = pipeline.run(data)

                formatted_res = format_chroma_resolution(resolution)
                filename_no_ext = path.name.rsplit(".", 1)[0]

                print(
                    f"Processed {path} for {formatted_res} chroma resolution"
                )

                processed.to_csv(
                    f"{output_filepath}/{filename_no_ext}_{formatted_res}.csv"
                )


def make_pipeline(chroma_res: float):
    pipeline = Pipeline()
    pipeline.add_task(ChromaResolution(chroma_res))
    pipeline.add_task(NormalizedChroma())

    group = PipelineTaskGroup()
    group.add_task(TemplateBased())
    group.add_task(Complexity())

    pipeline.add_task(group)
    pipeline.add_task(RemoveColumns(CHROMA_COLS))
    pipeline.add_task(MeanAndStd())

    return pipeline


if __name__ == "__main__":
    main()
