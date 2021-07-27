import click
import pathlib
import pandas as pd
from src.data.pipeline import Pipeline
from src.features.chroma_resolution import ChromaResolution
from src.features.normalized_chroma import NormalizedChroma
from src.utils.formatters import format_chroma_resolution

# Create datasets for each chroma resolution

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    COL_NAMES = ['piece', 'time', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12']
    RESOLUTIONS = [0.1, 0.5, 10, ChromaResolution.GLOBAL]

    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            data = pd.read_csv(path, header=None, names=COL_NAMES, dtype={'piece': str})
            data = data.fillna(method='ffill')

            for resolution in RESOLUTIONS:
                pipeline = make_pipeline(resolution)

                processed = pipeline.run(data)

                formatted_res = format_chroma_resolution(resolution)
                filename_no_ext = path.name.rsplit('.', 1)[0]

                print(f'Processed {path} for {formatted_res} chroma resolution')

                processed.to_csv(output_filepath + '/' + filename_no_ext + '_' + formatted_res + '.csv')

def make_pipeline(chroma_res: float):
    pipeline = Pipeline()
    pipeline.add_step(ChromaResolution(chroma_res))
    pipeline.add_step(NormalizedChroma())

    return pipeline

if __name__ == '__main__':
    main()