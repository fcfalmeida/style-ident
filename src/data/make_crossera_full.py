import click
import pathlib
import pandas as pd
from src.data.constants import CHROMA_COLS

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    COL_NAMES = ['piece', 'time']
    COL_NAMES.extend(CHROMA_COLS)

    crossera_full = pd.DataFrame(index=None)

    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            data = pd.read_csv(path, header=None, names=COL_NAMES, dtype={'piece': str})

            crossera_full = crossera_full.append(data)

    crossera_full.to_csv(output_filepath + '/' + 'chroma-nnls_full.csv', index=False)

if __name__ == '__main__':
    main()
    