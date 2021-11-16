import click
import pathlib
import pandas as pd
import numpy as np
import csv
from joblib import dump
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
)
from sklearn.metrics import plot_confusion_matrix
from src.data.constants.others import (
    PROCESSED_DIR, TRAINOUT_DIR, CONFUSION_MATRICES_DIR
)
from src.data.dataset_config import dataset_config
from src.data.constants.others import MODELS_DIR


@click.command()
@click.argument('dataset', type=str)
@click.argument('pipeline_name', type=str)
def main(dataset, pipeline_name):
    execute(dataset, pipeline_name)


def execute(dataset: str, pipeline_name: str):
    input_filepath = f'{PROCESSED_DIR}/{dataset}/{pipeline_name}'

    target_col = dataset_config[dataset]['target_col']

    print('-' * 75)
    print(f'Dataset -> {dataset}')
    print(f'Pipeline -> {pipeline_name}')
    print('-' * 75)

    trainout_filepath = f'{TRAINOUT_DIR}/{dataset}_trainout.csv'

    with open(trainout_filepath, 'a', newline='') as trainout_file:
        writer = csv.writer(trainout_file)

        data = pd.read_csv(
            f'{input_filepath}.csv', dtype={'piece': str}, index_col='piece')

        X = data.drop(target_col, axis=1).values
        y = data[target_col].values

        c = [2 ** x for x in range(-5, 17, 2)]
        gamma = [2 ** x for x in range(-15, 5, 2)]

        search_params = {
            'C': c,
            'gamma': gamma
        }
        svc = svm.SVC(kernel='rbf')

        cv = StratifiedKFold(n_splits=5, shuffle=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3
        )

        clf = GridSearchCV(svc, search_params, cv=cv)
        clf.fit(X_train, y_train)

        overall_mean_acc = 0
        interfold_std = 0
        mean_accuracies = []
        runs = 10
        for _ in range(runs):
            mean, std = _get_scores(clf, X_test, y_test)

            overall_mean_acc += mean
            mean_accuracies.append(mean)
            interfold_std += std

        overall_mean_acc /= runs
        interfold_std /= runs
        interrun_std = np.std(mean_accuracies)

        print(clf.best_params_)

        print(f'Mean Accuracy: {overall_mean_acc}')
        print(f'Inter-run deviation: {interrun_std}')
        print(f'Inter-fold deviation: {interfold_std}')

        writer.writerow([
            pipeline_name,
            overall_mean_acc,
            interrun_std,
            interfold_std
        ])

        _create_conf_matrix(
            clf,
            X_test,
            y_test,
            f'{CONFUSION_MATRICES_DIR}/{dataset}',
            pipeline_name
        )

        _save_model(clf, dataset, pipeline_name)

    print('-' * 75)


def _get_scores(clf, X_test, y_test):
    cv = StratifiedKFold(n_splits=3, shuffle=True)

    scores = cross_val_score(clf, X_test, y_test, cv=cv)

    return scores.mean(), scores.std()  # inter-fold deviation


def _create_conf_matrix(clf, X_test, y_test, path, pipeline_name):
    fig, ax = plt.subplots(figsize=(8, 8))

    plot_confusion_matrix(
        clf, X_test, y_test, normalize='true',
        values_format='.2%', cmap='Greys', colorbar=False, ax=ax)

    plt.xticks(rotation=45)

    pathlib.Path(path).mkdir(exist_ok=True)

    plt.savefig(f'{path}/{pipeline_name}.png')


def _save_model(clf, dataset, pipeline_name):
    output_filepath = f'{MODELS_DIR}/{dataset}'

    pathlib.Path(output_filepath).mkdir(exist_ok=True)

    dump(clf, f'{output_filepath}/{pipeline_name}')


if __name__ == '__main__':
    main()
