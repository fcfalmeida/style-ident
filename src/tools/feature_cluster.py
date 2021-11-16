import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from src.data.constants.others import INTERIM_DIR


@click.command()
@click.argument('dataset', type=str)
@click.argument('pipeline_name', type=str)
def main(dataset, pipeline_name):
    df = pd.read_csv(
        f'{INTERIM_DIR}/{dataset}/{pipeline_name}.csv', dtype={'piece': str},
        index_col='piece')

    X = df.values

    fig, ax = plt.subplots(figsize=(12, 8))
    corr = spearmanr(X).correlation
    corr = np.nan_to_num(corr)

    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    hierarchy.dendrogram(
        dist_linkage, labels=df.columns, ax=ax, leaf_rotation=90
    )

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
