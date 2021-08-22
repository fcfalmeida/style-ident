import click
import pathlib
import pandas as pd
import numpy as np
from src.data.constants import CHROMA_COLS

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    COL_NAMES = ['piece', 'time']
    COL_NAMES.extend(CHROMA_COLS)

    crossera_full = np.empty((0, len(COL_NAMES)))

    p = pathlib.Path(input_filepath)
    piano_files = list(p.glob('chroma-nnls_piano*'))
    orchestra_files = list(p.glob('chroma-nnls_orchestra*'))

    crossera_piano = join_datasets(piano_files, 'piano', COL_NAMES, output_filepath)
    crossera_orchestra = join_datasets(orchestra_files, 'orchestra', COL_NAMES, output_filepath)

    crossera_piano.to_csv(f'{output_filepath}/chroma-nnls_piano.csv', index=False)
    crossera_orchestra.to_csv(f'{output_filepath}/chroma-nnls_orchestra.csv', index=False)
    
    crossera_full = np.append(crossera_piano, crossera_orchestra, axis=0)
    df = pd.DataFrame(crossera_full, columns=COL_NAMES)
    df.to_csv(output_filepath + '/' + 'chroma-nnls_full.csv', index=False)

def join_datasets(files: list[str], type: str, columns: list[str], path: str):
    joined = np.empty((0, len(columns)))

    for f in files:
        data = pd.read_csv(f, header=None, names=columns, dtype={'piece': str})

        joined = np.append(joined, data.to_numpy(), axis=0)

        df = pd.DataFrame(joined, columns=columns)
        return df

if __name__ == '__main__':
    main()
    