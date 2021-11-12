import click
import pathlib
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.data.constants.others import INTERIM_DIR, PROCESSED_DIR
from src.data.dataset_config import dataset_config


@click.command()
@click.argument('dataset', type=str)
@click.argument("pipeline_name", type=str)
def main(dataset, pipeline_name):
    execute(dataset, pipeline_name)


def execute(dataset, pipeline_name):
    input_filepath = f'{INTERIM_DIR}/{dataset}'
    output_filepath = f'{PROCESSED_DIR}/{dataset}'

    pathlib.Path(output_filepath).mkdir(exist_ok=True)

    target_col = dataset_config[dataset]['target_col']

    df = pd.read_csv(
        f'{input_filepath}/{pipeline_name}.csv',
        dtype={"piece": str},
        index_col="piece"
    )

    df = _add_class_labels(df, target_col)

    lda = LinearDiscriminantAnalysis()

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    transformed = lda.fit_transform(X, y)

    transformed_df = pd.DataFrame(
        transformed, index=df.index
    )
    transformed_df = _add_class_labels(transformed_df, target_col)

    outfile = f'{output_filepath}/{pipeline_name}.csv'
    transformed_df.to_csv(outfile)

    print(f'Created {pipeline_name} trainset')


def _add_class_labels(data: pd.DataFrame, colname: str) -> pd.DataFrame:
    data[colname] = data.index.str.split("/").str[0]
    data[colname] = data[colname].str.split("_").str[-1]

    return data


if __name__ == '__main__':
    main()
