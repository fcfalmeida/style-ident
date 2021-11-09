import click
import pathlib
import pandas as pd
import numpy as np
import csv
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from src.data.constants.others import PROCESSED_DIR, TRAINOUT_DIR
from src.data.dataset_config import dataset_config


@click.command()
@click.argument('dataset', type=str)
@click.argument("pipeline_name", type=str)
def main(dataset, pipeline_name):
    execute(dataset, pipeline_name)


def execute(dataset: str, pipeline_name: str):
    input_filepath = f'{PROCESSED_DIR}/{dataset}/{pipeline_name}'

    target_col = dataset_config[dataset]['target_col']

    print('-' * 75)
    print(f'Pipeline: {pipeline_name}')
    print('-' * 75)

    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():

            with open(
                f'{TRAINOUT_DIR}/{path.stem}_trainout.csv', 'a', newline=''
            ) as trainout_file:
                writer = csv.writer(trainout_file)

                print(f"Dataset -> {path.name}")
                print("-" * 50)

                data = pd.read_csv(
                    path, dtype={"piece": str}, index_col="piece")

                le = LabelEncoder()
                X = data.drop(target_col, axis=1).values
                y = le.fit_transform(data[target_col])

                c = [2 ** x for x in range(-5, 17, 2)]
                gamma = [2 ** x for x in range(-15, 5, 2)]

                search_params = {
                    "C": c,
                    "gamma": gamma
                }
                svc = svm.SVC(kernel="rbf")

                cv = KFold(n_splits=5, shuffle=True)

                clf = GridSearchCV(svc, search_params, cv=cv)
                clf.fit(X, y)

                overall_mean_acc = 0
                interfold_std = 0
                mean_accuracies = []
                runs = 10
                for _ in range(runs):
                    mean, std = train(
                        X, y, clf.best_params_["C"], clf.best_params_["gamma"]
                    )

                    overall_mean_acc += mean
                    mean_accuracies.append(mean)
                    interfold_std += std

                overall_mean_acc /= runs
                interfold_std /= runs
                interrun_std = np.std(mean_accuracies)

                print(clf.best_params_)

                print(f"Mean Accuracy: {overall_mean_acc}")
                print(f"Inter-run deviation: {interrun_std}")
                print(f"Inter-fold deviation: {interfold_std}")
                print("-" * 50)

                writer.writerow([
                    pipeline_name,
                    overall_mean_acc,
                    interrun_std,
                    interfold_std
                ])

    print('-' * 75)
    print('-' * 75)


def train(X, y, C, gamma):
    clf = svm.SVC(kernel="rbf", C=C, gamma=gamma)
    cv = KFold(n_splits=3, shuffle=True)

    scores = cross_val_score(clf, X, y, cv=cv)

    return scores.mean(), scores.std()  # inter-fold deviation


if __name__ == "__main__":
    main()
