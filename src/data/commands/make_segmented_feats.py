import click
import pathlib
import pandas as pd
from src.data.pipelines.catalogue import segmented_pipelines
from src.data.constants.others import INTERIM_DIR, HCDF_SEGMENTED_DIR


@click.command()
@click.argument("pipeline_name", type=str)
def main(pipeline_name):
    execute(pipeline_name)


def execute(pipeline_name):
    input_filepath = HCDF_SEGMENTED_DIR
    output_filepath = f'{INTERIM_DIR}/{pipeline_name}'

    print(
        f'Processing segmented features for pipeline: {pipeline_name}'
    )

    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            data = pd.read_csv(path, dtype={'piece': str})
            data = data.fillna(method='ffill')
            data = data.set_index(['piece', 'time'])

            pipeline = segmented_pipelines[pipeline_name]()

            processed = pipeline.run(data)

            outfile = f'{output_filepath}/{path.name}'

            pathlib.Path(output_filepath).mkdir(exist_ok=True)
            processed.to_csv(outfile)

            print(f'Wrote {outfile}')


if __name__ == '__main__':
    main()
