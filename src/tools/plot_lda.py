import click
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.argument('filepath', type=click.Path(exists=True))
def main(filepath):
    df = pd.read_csv(filepath, dtype={'piece': str}, index_col='piece')

    lda = LinearDiscriminantAnalysis(n_components=2)

    le = LabelEncoder()
    X = df.drop('style_period', axis=1)
    y = le.fit_transform(df['style_period'])

    X_transformed = lda.fit_transform(X, y)

    transformed_df = pd.DataFrame(
        X_transformed, index=df.index, columns=['ld1', 'ld2']
    )
    transformed_df = _add_style_period_labels(transformed_df)

    plt.xlabel('Discriminant 1')
    plt.ylabel('Discriminant 2')

    baroque = transformed_df.loc[
        transformed_df['style_period'] == 'baroque'
    ]
    classical = transformed_df.loc[
        transformed_df['style_period'] == 'classical'
    ]
    romantic = transformed_df.loc[
        transformed_df['style_period'] == 'romantic'
    ]
    modern = transformed_df.loc[
        transformed_df['style_period'] == 'modern'
    ]

    alpha = 0.7
    baroque_scatter = plt.scatter(
        baroque['ld1'], baroque['ld2'], alpha=alpha
    )
    classical_scatter = plt.scatter(
        classical['ld1'], classical['ld2'], alpha=alpha
    )
    romantic_scatter = plt.scatter(
        romantic['ld1'], romantic['ld2'], alpha=alpha
    )
    modern_scatter = plt.scatter(
        modern['ld1'], modern['ld2'], alpha=alpha
    )

    plt.legend(
        (baroque_scatter, classical_scatter, romantic_scatter, modern_scatter),
        ('baroque', 'classical', 'romantic', 'modern'),
        fontsize=8
    )
    plt.show()


def _add_style_period_labels(data: pd.DataFrame) -> pd.DataFrame:
    data["style_period"] = data.index.str.split("/").str[0]
    data["style_period"] = data["style_period"].str.split("_").str[1]

    return data


if __name__ == '__main__':
    main()
