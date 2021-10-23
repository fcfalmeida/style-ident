import click
import pathlib
import pandas as pd
from src.data.pipelines import pipeline_catalogue


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("pipeline_name", type=str)
def main(input_filepath, output_filepath, pipeline_name):
    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            data = pd.read_csv(path, dtype={'piece': str})
            data = data.fillna(method='ffill')
            data = data.set_index(['piece', 'time'])

            pipeline = pipeline_catalogue[pipeline_name]

            processed = pipeline.run(data)

            print(f'Processed {path}')

            pathlib.Path(output_filepath).mkdir(exist_ok=True)
            processed.to_csv(f'{output_filepath}/{path.name}')


if __name__ == '__main__':
    main()
