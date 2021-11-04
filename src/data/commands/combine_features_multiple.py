import click
import src.data.commands.combine_pipeline_output as combine_pipeline_output
from src.data.constants.others import FEATURE_COMB_FILE


@click.command()
def main():
    pipelines = read_pipeline_list()

    for p in pipelines:
        parts = p.split()

        if len(parts) > 1:
            combine_pipeline_output.execute(parts)

            print(f'Combined features: {p}')


def read_pipeline_list():
    with open(FEATURE_COMB_FILE, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
    main()
