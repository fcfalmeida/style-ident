import click
import re
import pandas as pd
from matplotlib import pyplot as plt
from TIVlib import TIVCollection
from src.data.dataset_config import dataset_config
from src.data.constants.others import EXTERNAL_DIR, PROCESSED_DIR
from src.data.constants.features import TISFeats
from src.data.constants.feature_groups import CHROMA_FEATS
from src.features.tis_horizontal import TISHorizontal


@click.command()
@click.argument('dataset', type=str)
@click.argument('pipeline_name', type=str)
@click.argument('feature', type=str)
@click.argument('stat', type=str)
@click.argument('res', type=str)
def main(dataset, pipeline_name, feature, stat, res):
    input_filepath = f'{PROCESSED_DIR}/{dataset}/{pipeline_name}'

    df = pd.read_csv(
            f'{input_filepath}.csv', dtype={'piece': str})

    target_col = dataset_config[dataset]['target_col']
    classes = dataset_config[dataset]['classes']

    df[target_col] = pd.Categorical(df[target_col], classes)
    df = df.sort_values(by=[target_col, 'piece'])
    df = df.set_index('piece')

    fig, ax = plt.subplots()

    ax.set_xticklabels(classes)

    col_names = [col for col in df.columns if feature in col and stat in col and res in col]

    dfs_by_class = _split_dfs_by_class(df, target_col, classes)

    for c in col_names:
        y = []
        for i, class_data in enumerate(dfs_by_class):
            y.append(class_data[c].values)

        ax.boxplot(y)

    plt.show()


def _add_class_labels(data: pd.DataFrame, colname: str) -> pd.DataFrame:
    data[colname] = data.index.str.split("/").str[0]
    data[colname] = data[colname].str.split("_").str[-1]

    return data


def _make_xticks(count_per_class):
    xticks = [0]

    for i in range(len(count_per_class) - 1):
        count = count_per_class[i]
        xticks.append(xticks[i] + count)

    return xticks


def _split_dfs_by_class(df, target_col, classes):
    dfs_by_class = []

    for class_name in classes:
        data = df.loc[df[target_col] == class_name]
        dfs_by_class.append(data)

    return dfs_by_class


def _format_label(label):
    res = re.search(r'\w+', label).group(0)

    return res


if __name__ == '__main__':
    main()
