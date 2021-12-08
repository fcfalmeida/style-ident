import click
import pandas as pd
from matplotlib import pyplot as plt
from TIVlib import TIV, TIVCollection
from src.data.constants.others import EXTERNAL_DIR
from src.data.constants.feature_groups import CHROMA_FEATS


@click.command()
@click.argument('dataset', type=str)
@click.argument('piece', type=str)
@click.argument('coef', type=int)
def main(dataset, piece, coef):
    df = pd.read_csv(f'{EXTERNAL_DIR}/{dataset}.csv', dtype={"piece": str})
    df = df.fillna(method='ffill')
    df = df.set_index(['piece', 'time'])

    piece_df = df.loc[piece]

    tivs = TIVCollection.from_pcp(piece_df[CHROMA_FEATS].values.T)
    diatonicity_vectors = tivs.vectors[:, coef]
    diatonicity_vectors = diatonicity_vectors / TIV.weights[coef]

    piece_tiv = TIV.from_pcp(piece_df[CHROMA_FEATS].sum().values)
    piece_diatonicity_vector = piece_tiv.vector[coef]
    piece_diatonicity_vector = piece_diatonicity_vector / TIV.weights[coef]

    _, ax = plt.subplots()

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid()

    ax.scatter(diatonicity_vectors.real, diatonicity_vectors.imag)
    ax.scatter(
        piece_diatonicity_vector.real, piece_diatonicity_vector.imag, c='red'
    )

    circle = plt.Circle((0, 0), 1, fill=False)
    ax.add_patch(circle)

    plt.show()


if __name__ == '__main__':
    main()
