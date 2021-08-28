import click
import pathlib
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from src.data.constants import CHROMA_COLS

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    COL_NAMES = ['piece', 'time']
    COL_NAMES.extend(CHROMA_COLS)

    TYPES = ['piano', 'orchestra', 'full']

    p = pathlib.Path(input_filepath)

    for t in TYPES:
        files = list(p.glob(f'chroma-nnls_{t}*'))

        df = join_datasets(files)
        df = _add_style_period_labels(df)

        lda = LinearDiscriminantAnalysis(n_components=3)

        le = LabelEncoder()
        X = df.drop('style_period', axis=1)
        y = le.fit_transform(df['style_period'])

        transformed = lda.fit_transform(X, y)

        transformed_df = pd.DataFrame(transformed, index=df.index, columns=['ld1', 'ld2', 'ld3'])
        transformed_df = _add_style_period_labels(transformed_df)

        transformed_df.to_csv(f'{output_filepath}/chroma-nnls_{t}.csv')

def _add_style_period_labels(data: pd.DataFrame) -> pd.DataFrame:
    data['style_period'] = data.index.str.split('/').str[0]
    data['style_period'] = data['style_period'].str.split('_').str[1]

    return data

def join_datasets(files: list[str]):
    joined = None

    for i, f in enumerate(files):
        data = pd.read_csv(f, dtype={'piece': str}, index_col='piece')

        resolution = f.name.split('_')[2]
        resolution = resolution.replace('.csv', '')

        data = data.add_suffix(f'_{resolution}')

        # Either initialize the dataset with the first one that is imported or join it with another
        if i == 0:
            joined = data.copy()
        else:
            joined = joined.join(data)

    return joined

if __name__ == '__main__':
    main()