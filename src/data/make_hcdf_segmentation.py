import click
import pathlib
import pandas as pd
from src.data.pipelines.pipeline import Pipeline
from src.features.hcdf_segmentation import HCDFSegmentation


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
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
