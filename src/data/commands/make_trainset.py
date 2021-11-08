import click
import pathlib
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from src.data.constants.others import INTERIM_DIR, PROCESSED_DIR
from src.data.dataset_config import dataset_config


@click.command()
@click.argument('dataset', type=str)
@click.argument("pipeline_name", type=str)
def main(dataset, pipeline_name):
    execute(dataset, pipeline_name)


def execute(dataset, pipeline_name):
    input_filepath = f'{INTERIM_DIR}/{dataset}/{pipeline_name}'
    output_filepath = f'{PROCESSED_DIR}/{dataset}/{pipeline_name}'

    target_col = dataset_config[dataset]['target_col']

    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            df = pd.read_csv(path, dtype={"piece": str}, index_col="piece")

            df = _add_class_labels(df, target_col)

            n_components = len(dataset_config[dataset]['classes']) - 1

            lda = LinearDiscriminantAnalysis(n_components=n_components)

            le = LabelEncoder()
            X = df.drop(target_col, axis=1)
            y = le.fit_transform(df[target_col])

            transformed = lda.fit_transform(X, y)

            transformed_df = pd.DataFrame(
                transformed, index=df.index
            )
            transformed_df = _add_class_labels(transformed_df, target_col)

            pathlib.Path(output_filepath).mkdir(exist_ok=True)
            transformed_df.to_csv(f"{output_filepath}/{path.name}")


def _add_class_labels(data: pd.DataFrame, colname: str) -> pd.DataFrame:
    data[colname] = data.index.str.split("/").str[0]
    data[colname] = data[colname].str.split("_").str[-1]

    return data


if __name__ == '__main__':
    main()
