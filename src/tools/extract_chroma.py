import click
import librosa
import vamp
import pathlib
import pandas as pd
from src.data.constants.others import AUDIO_DIR, EXTERNAL_DIR
from src.data.constants.feature_groups import CHROMA_FEATS


@click.command()
@click.argument('dataset', type=str)
def main(dataset):
    RESOLUTION = 0.1

    input_filepath = f'{AUDIO_DIR}/{dataset}'
    output_filepath = f'{EXTERNAL_DIR}/{dataset}'

    COL_NAMES = ['piece', 'time']
    COL_NAMES.extend(CHROMA_FEATS)

    rows = []

    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            y, sr = librosa.load(path)

            chroma = _get_nnls(y, sr, 2048, 2048)

            for i, chroma_vector in enumerate(chroma):
                row = {'piece': path.stem, 'time': (i+1) * RESOLUTION}

                for j, c in enumerate(chroma_vector):
                    row[CHROMA_FEATS[j]] = c

                rows.append(row)

    df = pd.DataFrame(rows, columns=COL_NAMES)
    df = df.set_index(['piece', 'time'])

    df.to_csv(f'{output_filepath}/{dataset}.csv')

    print(df)


def _get_nnls(y, sr, fr, off):
    """
        returns nnls chromagram
        Parameters
        ----------
        y : number > 0 [scalar]
            audio
        sr: number > 0 [scalar]
            chroma-samplerate-framesize-overlap
        fr: number [scalar]
            frame size of windos
        off: number [scalar]
            overlap
        Returns
        -------
        list of chromagrams
    """
    plugin = 'nnls-chroma:nnls-chroma'
    chroma = list(vamp.process_audio(y, sr, plugin, output="chroma", block_size=fr, step_size=off))
    vectors = []

    for c in chroma:
        vectors.append(c['values'].tolist())

    return vectors


if __name__ == "__main__":
    main()
