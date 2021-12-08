import click
import pathlib
import pandas as pd
import numpy as np
import csv
from joblib import dump
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, StratifiedGroupKFold
)
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from src.data.constants.others import (
    ANNOTATIONS_DIR, PROCESSED_DIR, TRAINOUT_DIR, CONFUSION_MATRICES_DIR
)
from src.data.dataset_config import dataset_config
from src.data.constants.others import MODELS_DIR


@click.command()
@click.argument('dataset', type=str)
@click.argument('pipeline_name', type=str)
@click.option('--composer-filter', '-cf', is_flag=True)
def main(dataset, pipeline_name, composer_filter):
    print(composer_filter)
    execute(dataset, pipeline_name, composer_filter)


def execute(dataset: str, pipeline_name: str, composer_filter: bool):
    input_filepath = f'{PROCESSED_DIR}/{dataset}/{pipeline_name}'

    target_col = dataset_config[dataset]['target_col']

    print('-' * 75)
    print(f'Dataset -> {dataset}')
    print(f'Pipeline -> {pipeline_name}')
    print('-' * 75)

    trainout_filepath = f'{TRAINOUT_DIR}/{dataset}_\
        {"filter_" if composer_filter else ""}trainout.csv'

    with open(trainout_filepath, 'a', newline='') as trainout_file:
        writer = csv.writer(trainout_file)

        data = pd.read_csv(
            f'{input_filepath}.csv', dtype={'piece': str}, index_col='piece')

        X = data.drop(target_col, axis=1).values
        y = data[target_col].values

        overall_mean_acc = 0
        run_mean_accuracy_values = []
        inter_fold_dev = 0
        runs = 10

        for _ in range(runs):
            fold_mean_accuracy_values = []

            split = _get_cv_split(X, y, composer_filter, dataset)

            for train_index, test_index in split:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                X_train_transformed, X_test_transformed = lda_transform(
                    X_train, y_train, X_test
                )

                clf = train_classifier(X_train_transformed, y_train)

                acc = prediction_acc(clf, X_test_transformed, y_test)

                print(clf.best_params_)
                print(acc)

                overall_mean_acc += acc

                fold_mean_accuracy_values.append(acc)

            mean_run_accuracy = np.mean(fold_mean_accuracy_values)
            run_mean_accuracy_values.append(mean_run_accuracy)

            inter_fold_dev += np.std(fold_mean_accuracy_values)

        inter_run_dev = np.std(run_mean_accuracy_values)
        inter_fold_dev /= runs

        overall_mean_acc /= (runs * 3)

        print(f'Mean Accuracy: {overall_mean_acc}')
        print(f'Inter-run Deviation: {inter_run_dev}')
        print(f'Inter-fold Deviation: {inter_fold_dev}')

        writer.writerow([
            pipeline_name,
            overall_mean_acc
        ])

        _create_conf_matrix(
            clf,
            X_test_transformed,
            y_test,
            f'{CONFUSION_MATRICES_DIR}/{dataset}',
            pipeline_name
        )

        _save_model(clf, dataset, pipeline_name)

    print('-' * 75)


def _get_cv_split(X, y, composer_filter, dataset):
    split_cv = None
    groups = None

    filter_col = dataset_config[dataset]['filter_col']

    if composer_filter:
        split_cv = StratifiedGroupKFold(n_splits=3, shuffle=True)

        annotations = pd.read_csv(f'{ANNOTATIONS_DIR}/{dataset}.csv')
        # Trim groups to exclude addons
        groups = annotations[filter_col].values[:X.shape[0]]
    else:
        split_cv = StratifiedKFold(n_splits=3, shuffle=True)

    split = split_cv.split(X, y, groups)

    return split


def lda_transform(X_train, y_train, X_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    X_train_transformed = lda.transform(X_train)
    X_test_transformed = lda.transform(X_test)

    return X_train_transformed, X_test_transformed


def train_classifier(X_train, y_train):
    c = [2 ** x for x in range(-5, 17, 2)]
    gamma = [2 ** x for x in range(-15, 5, 2)]

    kernel = ['linear', 'poly']
    degree = [2, 3, 4]

    search_params = {
        'C': c
    }

    clf = svm.SVC(kernel='linear')

    gs_cv = StratifiedKFold(n_splits=5, shuffle=True)
    clf = GridSearchCV(clf, search_params, cv=gs_cv)

    clf.fit(X_train, y_train)

    return clf


def prediction_acc(clf, X_test, y_true):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_true, y_pred)

    return acc


def _create_conf_matrix(clf, X_test, y_test, path, pipeline_name):
    _, ax = plt.subplots(figsize=(8, 8))

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
