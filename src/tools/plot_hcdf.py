import click
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from lib.HCDF import HCDF
from src.data.constants import CHROMA_COLS


@click.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.argument('piece', type=str)
def main(filepath, piece):
    CHROMA_RESOLUTION = 0.1

    df = pd.read_csv(filepath, dtype={'piece': str})
    df = df.fillna(method='ffill')

    peak_indexes, peak_mags, hcdf = HCDF.harmonic_change(
        df.loc[df['piece'] == piece][CHROMA_COLS].values
    )

    fig, ax = plt.subplots()
    fig.canvas.set_window_title(f'HCDF - {piece}')

    ax.set(xlabel='Time (s)', ylabel='HCDF magnitude')

    hcdf_x = np.arange(0, hcdf.size) * CHROMA_RESOLUTION
    ax.plot(hcdf_x, hcdf, label='hcdf values')

    peak_indexes_x = peak_indexes * CHROMA_RESOLUTION
    ax.scatter(
        peak_indexes_x, peak_mags, label='hcdf peaks', color='red', s=10
    )

    ax.xaxis.set_ticks(np.arange(min(hcdf_x), max(hcdf_x) + 1, 5))
    ax.grid(ls='dashed')
    
    plt.show()


if __name__ == '__main__':
    main()
