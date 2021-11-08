import click
import pathlib
import pandas as pd
from src.features.chroma_resolution import ChromaResolution
from src.utils.formatters import format_chroma_resolution
from src.data.pipelines.catalogue import res_pipelines
from src.data.constants.others import EXTERNAL_DIR, INTERIM_DIR
from src.data.dataset_config import dataset_config


@click.command()
@click.argument("dataset", type=str)
@click.argument("pipeline_name", type=str)
def main(dataset, pipeline_name):
    execute(dataset, pipeline_name)


def execute(dataset, pipeline_name):
    RESOLUTIONS = [0.1, 0.5, 10, ChromaResolution.GLOBAL]

    input_filepath = f'{EXTERNAL_DIR}/{dataset}'
    output_filepath = f'{INTERIM_DIR}/{dataset}/{pipeline_name}'

    print(
        f'Processing resolution features for pipeline: {pipeline_name}'
    )

    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            data = pd.read_csv(path, dtype={"piece": str})
            data = data.fillna(method="ffill")

            for resolution in RESOLUTIONS:
                pipeline = res_pipelines[pipeline_name](resolution)

                processed = pipeline.run(data)

                formatted_res = format_chroma_resolution(resolution)
                filename_no_ext = path.name.rsplit(".", 1)[0]

                outfile = \
                    f'{output_filepath}/{filename_no_ext}_{formatted_res}.csv'

                pathlib.Path(output_filepath).mkdir(exist_ok=True)
                processed.to_csv(outfile)

                print(f'Wrote {outfile}')

    for cat in dataset_config[dataset]['categories']:
        files = list(
            pathlib.Path(output_filepath).glob(f"chroma-nnls_{cat}_*")
        )

        joined = join_datasets(files)

        joined.to_csv(f"{output_filepath}/chroma-nnls_{cat}.csv")

    cleanup_output_dir(output_filepath)


def join_datasets(files: list[str]):
    joined = None

    for f in files:
        data = pd.read_csv(f, dtype={"piece": str}, index_col="piece")

        resolution = f.name.split("_")[-1]
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
