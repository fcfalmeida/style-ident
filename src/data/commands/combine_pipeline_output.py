import click
import pathlib
from src.data.pipeline_output_combiner import PipelineOutputCombiner
from src.data.constants.others import INTERIM_DIR
from src.data.dataset_config import dataset_config


@click.command()
@click.argument('dataset', type=str)
@click.argument('pipelines', type=str, nargs=-1)
def main(dataset, pipelines):
    execute(dataset, pipelines)


def execute(dataset, pipelines):
    pipelines_str = '_'.join(pipelines)

    for cat in dataset_config[dataset]['categories']:
        data = PipelineOutputCombiner.combine(pipelines, cat, 'piece')

        output_dir = f'{INTERIM_DIR}/{dataset}/{pipelines_str}'
        pathlib.Path(output_dir).mkdir(exist_ok=True)

        data.to_csv(f'{output_dir}/chroma-nnls_{cat}.csv')


if __name__ == '__main__':
    main()
