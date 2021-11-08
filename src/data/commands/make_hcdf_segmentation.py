import click
import pathlib
import pandas as pd
from src.data.pipelines.pipeline import Pipeline
from src.features.hcdf_segmentation import HCDFSegmentation
from src.data.constants.others import EXTERNAL_DIR, HCDF_SEGMENTED_DIR


@click.command()
@click.argument('dataset', type=str)
def main(dataset):
    input_filepath = f'{EXTERNAL_DIR}/{dataset}'
    output_filepath = f'{HCDF_SEGMENTED_DIR}/{dataset}'

    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            data = pd.read_csv(path, dtype={"piece": str})
            data = data.fillna(method="ffill")
            data = data.set_index(['piece', 'time'])

            pipeline = make_pipeline()

            processed = pipeline.run(data)

            print(f"Processed {path}")

            processed.to_csv(f"{output_filepath}/{path.name}")


def make_pipeline():
    pipeline = Pipeline()

    pipeline.add_task(HCDFSegmentation())

    return pipeline


if __name__ == "__main__":
    main()
