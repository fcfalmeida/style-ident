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

    plt.xlabel('Discriminant 1')
    plt.ylabel('Discriminant 2')
    plt.scatter(
        X_transformed[:, 0],
        X_transformed[:, 1],
        c=y,
        cmap='rainbow',
        alpha=0.7,
        edgecolors='b'
    )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
