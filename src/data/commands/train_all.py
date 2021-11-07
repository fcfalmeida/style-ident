import click
import src.models.weiss as weiss
from src.data.pipelines.catalogue import all_pipeline_combinations


@click.command()
def main():
    for pipeline_comb in all_pipeline_combinations():
        pipeline_name = '_'.join(pipeline_comb)
        weiss.execute(pipeline_name)


if __name__ == '__main__':
    main()
