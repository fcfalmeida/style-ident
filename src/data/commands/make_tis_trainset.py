import click
import pathlib
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            df = pd.read_csv(path, dtype={"piece": str}, index_col="piece")

            df = _add_style_period_labels(df)

            lda = LinearDiscriminantAnalysis(n_components=3)

            le = LabelEncoder()
            X = df.drop("style_period", axis=1)
            y = le.fit_transform(df["style_period"])

            transformed = lda.fit_transform(X, y)

            transformed_df = pd.DataFrame(
                transformed, index=df.index, columns=["ld1", "ld2", "ld3"]
            )
            transformed_df = _add_style_period_labels(transformed_df)

            transformed_df.to_csv(f"{output_filepath}/{path.name}")


def _add_style_period_labels(data: pd.DataFrame) -> pd.DataFrame:
    data["style_period"] = data.index.str.split("/").str[0]
    data["style_period"] = data["style_period"].str.split("_").str[1]

    return data


if __name__ == '__main__':
    main()
