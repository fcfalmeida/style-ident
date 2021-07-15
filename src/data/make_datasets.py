import click
import pandas as pd
from src.data.helpers import aggregate_chroma

#for i, row in data.iterrows():
#    print(i)
    #exchange = row['exchange']

# Create datasets for each chroma resolution

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    print('Make datasets')
    print(input_filepath)
    print(output_filepath)

    COL_NAMES = ['piece', 'time', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12']

    data = pd.read_csv('data/external/chroma-nnls_orchestra_baroque.csv', header=None, names=COL_NAMES)
    data.fillna(method='ffill', inplace=True)

    print(data)

    print(aggregate_chroma(data, 10, 10))

if __name__ == '__main__':
    main()