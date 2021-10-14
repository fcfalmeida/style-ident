import click
import pathlib
import pandas as pd
from src.data.pipeline import Pipeline
from src.data.pipeline_task_group import PipelineTaskGroup
from src.features.tis import TIS
from src.data.remove_chroma import RemoveChroma
from src.features.mean_and_std import MeanAndStd
from src.features.median_and_iqr import MedianAndIQR


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

    pipeline.add_task(RemoveChroma())
    pipeline.add_task(MeanAndStd())
    pipeline.add_task(MedianAndIQR())

    return pipeline


if __name__ == '__main__':
    main()
