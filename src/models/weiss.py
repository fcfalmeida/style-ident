import click
import pathlib
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, ShuffleSplit


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    for path in pathlib.Path(input_filepath).iterdir():
        if path.is_file():
            print(f"Dataset -> {path.name}")
            print("-" * 50)

            data = pd.read_csv(path, dtype={"piece": str}, index_col="piece")

            le = LabelEncoder()
            X = data.drop("style_period", axis=1)
            y = le.fit_transform(data["style_period"])

            c = [2 ** x for x in range(-5, 15)]

            search_params = {
                "C": c,
                "gamma": ["scale", "auto"]
            }
            svc = svm.SVC(kernel="rbf")

            cv = ShuffleSplit(n_splits=5, test_size=1 / 5)

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


def train(X, y, C, gamma):
    clf = svm.SVC(kernel="rbf", C=C, gamma=gamma)
    cv = ShuffleSplit(n_splits=3, test_size=1 / 3)

    scores = cross_val_score(clf, X, y, cv=cv)

    return scores.mean(), scores.std()  # inter-fold deviation


if __name__ == "__main__":
    main()
