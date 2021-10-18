import click
import pathlib
import pandas as pd
from src.data.pipeline import Pipeline
from src.data.pipeline_task_group import PipelineTaskGroup
from src.data.constants import CHROMA_COLS, HCDF_PEAK_IDX
from src.features.tis import TIS
from src.data.remove_columns import RemoveColumns
from src.features.mean_and_std import MeanAndStd


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            data = pd.read_csv(path, dtype={'piece': str})
            data = data.fillna(method='ffill')
            data = data.set_index(['piece', 'time'])

            pipeline = make_pipeline()

            processed = pipeline.run(data)

            print(f'Processed {path}')

            processed.to_csv(f'{output_filepath}/{path.name}')


def make_pipeline():
    pipeline = Pipeline()

    group = PipelineTaskGroup()
    group.add_task(TIS())
    pipeline.add_task(group)

    remove_cols = CHROMA_COLS.copy()
    remove_cols.append(HCDF_PEAK_IDX)

    pipeline.add_task(RemoveColumns(remove_cols))
    pipeline.add_task(MeanAndStd())

    return pipeline


if __name__ == '__main__':
    main()
