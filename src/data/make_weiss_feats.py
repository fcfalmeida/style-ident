import click
import pathlib
import pandas as pd
from src.features.chroma_resolution import ChromaResolution
from src.utils.formatters import format_chroma_resolution
from src.data.pipelines import pipeline_catalogue


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("pipeline_name", type=str)
def main(input_filepath, output_filepath, pipeline_name):
    RESOLUTIONS = [0.1, 0.5, 10, ChromaResolution.GLOBAL]

    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            data = pd.read_csv(path, dtype={"piece": str})
            data = data.fillna(method="ffill")

            for resolution in RESOLUTIONS:
                pipeline = pipeline_catalogue[pipeline_name](resolution)

                processed = pipeline.run(data)

                formatted_res = format_chroma_resolution(resolution)
                filename_no_ext = path.name.rsplit(".", 1)[0]

                print(
                    f"Processed {path} for {formatted_res} chroma resolution"
                )

                pathlib.Path(output_filepath).mkdir(exist_ok=True)
                processed.to_csv(
                    f"{output_filepath}/{filename_no_ext}_{formatted_res}.csv"
                )

    TYPES = ["piano", "orchestra", "full"]

    for t in TYPES:
        files = list(pathlib.Path(output_filepath).glob(f"chroma-nnls_{t}_*"))

        joined = join_datasets(files)

        joined.to_csv(f"{output_filepath}/chroma-nnls_{t}.csv")

    cleanup_output_dir(output_filepath)


def join_datasets(files: list[str]):
    joined = None

    for f in files:
        data = pd.read_csv(f, dtype={"piece": str}, index_col="piece")

        resolution = f.name.split("_")[2]
        resolution = resolution.replace(".csv", "")

        data = data.add_suffix(f"_{resolution}")

        if joined is None:
            joined = data.copy()
        else:
            joined = joined.join(data)

    return joined


def cleanup_output_dir(output_dir):
    path = pathlib.Path(output_dir).glob('chroma-nnls_*_*')

    for file in path:
        file.unlink()


if __name__ == "__main__":
    main()
