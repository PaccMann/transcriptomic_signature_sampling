import os
import argparse
from typing import Iterable
import joblib
import numpy as np
import pandas as pd
import sklearn.metrics as sm
from sklearn.metrics import roc_auc_score
from signature_sampling.hyperparameter_factory import CLASSIFIER_FACTORY
from typing import Iterable
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()


def run_clf(
    model: str,
    data_dir: str,
    result_root_dir: str,
    sampling_methods: Iterable,
    size: str,
    label: str = "cms",
    folds: int = 5,
    repeats: int = 5,
    cptac: bool = True,
    **kwargs,
) -> None:
    """Saves results from classification experiment in csv format in the designated folder.

    Args:
        model (str): Classification model to use. Refer hyperparameter factory for options.
        data_dir (str): Path to the data folder.
        result_root_dir (str): Path to where the results should be saved.
        sampling_methods (Iterable): List of data augmentation methods to compare.
        size (str): Class size upto which augmentation is performed.
        label (str, optional): The clinical label to use for the classification task.
            Defaults to "cms".
        folds (int, optional): Number of cross validation folds. Defaults to 5.
        repeats (int, optional): Number of cross validation repetitions. Defaults to 5.
        cptac (bool, optional): Whether to run the experiment on cptac data. Defaults to True.
    """
    result_dir = os.path.join(result_root_dir, model)
    os.makedirs(os.path.join(result_dir, "summary_results"), exist_ok=True)
    mean_scores = []
    mean_acc = []
    cptac_mean_scores = []
    cptac_mean_acc = []
    size_copy = size
    mean_metrics = pd.DataFrame(
        columns=["roc-auc", "bal_acc", "cptac_roc-auc", "cptac_bal_acc"],
        index=sampling_methods,
    )

    for sampling in sampling_methods:
        scores = []
        acc = []
        cptac_scores = []
        cptac_acc = []
        metrics = pd.DataFrame(
            columns=["roc-auc", "bal_acc", "cptac_roc-auc", "cptac_bal_acc"]
        )

        if sampling == "unbalanced":
            size = ""

        save_dir = os.path.join(result_dir, f"{sampling}", size)

        for i in range(1, (folds * repeats) + 1):
            os.makedirs(os.path.join(save_dir, f"split_{i}"), exist_ok=True)
            train_df = pd.read_csv(
                os.path.join(
                    data_dir,
                    f"{sampling}",
                    size,
                    f"{i}/train_logfpkm_colotype_stdz.csv",
                ),
                index_col=0,
            )
            train_labels = pd.read_csv(
                os.path.join(
                    data_dir,
                    f"{sampling}",
                    size,
                    f"{i}/train_labels_logfpkm_colotype.csv",
                ),
                index_col=0,
            )
            test_df = pd.read_csv(
                os.path.join(
                    data_dir, f"{sampling}", size, f"{i}/test_logfpkm_colotype_stdz.csv"
                ),
                index_col=0,
            )
            test_labels = pd.read_csv(
                os.path.join(
                    data_dir,
                    f"{sampling}",
                    size,
                    f"{i}/test_labels_logfpkm_colotype.csv",
                ),
                index_col=0,
            )

            y_train = train_labels[label]
            y_test = test_labels[label]

            if any(pd.isna(y_train)):
                drop_idx = np.argwhere(pd.isna(y_train.values))
                drop_idx = drop_idx.flatten()
                train_df = train_df.drop(index=train_df.index[drop_idx])
                y_train = y_train.drop(index=y_train.index[drop_idx])

            clf = CLASSIFIER_FACTORY[model]
            clf.fit(train_df, y_train)

            joblib.dump(clf, os.path.join(save_dir, f"split_{i}/{model}_model"))

            # TEST
            auc = roc_auc_score(y_test, clf.predict_proba(test_df), multi_class="ovr")
            bal_acc = sm.balanced_accuracy_score(y_test, clf.predict(test_df))

            scores.append(auc)
            acc.append(bal_acc)

            if cptac:
                cptac_label = str.upper(label)
                cptac_df = pd.read_csv(
                    os.path.join(data_dir, f"{sampling}", size, f"{i}/cptac_stdz.csv"),
                    index_col=0,
                )
                cptac_labels = pd.read_csv(
                    os.path.join(
                        data_dir, f"{sampling}", size, f"{i}/cptac_labels.csv"
                    ),
                    index_col=0,
                )

                cptac_preds = clf.predict(cptac_df)
                cptac_prob_preds = clf.predict_proba(cptac_df)
                # CPTAC
                drop_idx = np.argwhere(pd.isna(cptac_labels[cptac_label].values))
                keep_idx = np.setdiff1d(np.arange(len(cptac_preds)), drop_idx)

                cptac_pred_mod = cptac_preds[keep_idx]
                cptac_auc = roc_auc_score(
                    cptac_labels[cptac_label][keep_idx],
                    cptac_prob_preds[keep_idx],
                    multi_class="ovr",
                )
                cptac_bal_acc = sm.balanced_accuracy_score(
                    cptac_labels[cptac_label][keep_idx], cptac_pred_mod
                )
                cptac_scores.append(cptac_auc)
                cptac_acc.append(cptac_bal_acc)

        metrics["roc-auc"] = scores
        metrics["bal_acc"] = acc
        if cptac:
            metrics["cptac_roc-auc"] = cptac_scores
            metrics["cptac_bal_acc"] = cptac_acc
            cptac_mean_scores.append(np.mean(cptac_scores))
            cptac_mean_acc.append(np.mean(cptac_acc))

        metrics.to_csv(os.path.join(result_dir, f"{sampling}", size, "metrics.csv"))

        mean_scores.append(np.mean(scores))
        mean_acc.append(np.mean(acc))

    mean_metrics["roc-auc"] = mean_scores
    mean_metrics["bal_acc"] = mean_acc
    if cptac:
        mean_metrics["cptac_roc-auc"] = cptac_mean_scores
        mean_metrics["cptac_bal_acc"] = cptac_mean_acc

    mean_metrics.to_csv(
        os.path.join(result_dir, f"summary_results/mean_metrics_{size_copy}.csv")
    )


parser.add_argument(
    "data_dir", type=str, help="Path to the main directory containing the datasets."
)
parser.add_argument(
    "result_root_dir",
    type=str,
    help="Path to the directory where results will be saved.",
)


def main(data_dir: str, result_root_dir: str):
    sampling_methods = [
        "poisson_local",
        "gamma_poisson",
        "local_crossover",
        "global_crossover",
        "smote",
        "replacement",
        "unbalanced",
    ]
    for model in ["RF", "Logistic", "SVM-RBF", "KNN", "EBM"]:
        for size in ["max", "500", "5000"]:
            run_clf(model, data_dir, result_root_dir, sampling_methods, size)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.data_dir, args.result_root_dir)
