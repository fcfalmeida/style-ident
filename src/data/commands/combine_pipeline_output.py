import click
import pathlib
from src.data.pipeline_output_combiner import PipelineOutputCombiner
from src.data.constants.others import INTERIM_DIR


@click.command()
@click.argument('pipelines', type=str, nargs=-1)
def main(pipelines):
    TYPES = ['orchestra', 'piano', 'full']
    pipelines_str = '_'.join(pipelines)

    for t in TYPES:
        data = PipelineOutputCombiner.combine(pipelines, t, 'piece')

        output_dir = f'{INTERIM_DIR}/{pipelines_str}'
        pathlib.Path(output_dir).mkdir(exist_ok=True)

        data.to_csv(f'{output_dir}/chroma-nnls_{t}.csv')


if __name__ == '__main__':
    main()
