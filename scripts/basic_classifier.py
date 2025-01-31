import argparse
import json
import os
import warnings
from ast import literal_eval
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from signature_sampling.hyperparameter_factory import CLASSIFIER_FACTORY
from signature_sampling.torch_data import TCGADataset

warnings.filterwarnings("ignore")
seed = 42
parser = argparse.ArgumentParser()
np.random.seed(seed)
torch.manual_seed(seed)


def run_clf(
    model: str,
    data_dir: Path,
    result_root_dir: Path,
    sampling_methods: Iterable,
    params_path: Path,
    size: str,
    label: str = "PAM50",
    folds: int = 5,
    repeats: int = 5,
    external: str = "tcga_ext",
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
        external (bool, optional): Whether to run the experiment on external data. Defaults to True.
    """
    result_dir = os.path.join(result_root_dir, model)
    os.makedirs(os.path.join(result_dir, "summary_results"), exist_ok=True)

    with open(params_path, "r") as f:
        params = json.load(f)

    if "probability" in params[model].keys():
        params[model]["probability"] = literal_eval(params[model]["probability"])

    mean_scores = []
    std_scores = []
    mean_acc = []
    std_acc = []
    external_mean_scores = []
    external_std_scores = []
    external_mean_acc = []
    external_std_acc = []
    size_copy = size
    mean_metrics = pd.DataFrame(
        columns=["roc-auc", "bal_acc", "external_roc-auc", "external_bal_acc"],
        index=sampling_methods,
    )

    for sampling in sampling_methods:
        scores = []
        acc = []
        external_scores = []
        external_acc = []
        metrics = pd.DataFrame(
            columns=["roc-auc", "bal_acc", "external_roc-auc", "external_bal_acc"]
        )

        if sampling == "unaugmented":
            size = ""

        save_dir = os.path.join(result_dir, f"{sampling}", size)

        for i in range(1, (folds * repeats) + 1):
            os.makedirs(os.path.join(save_dir, f"split_{i}"), exist_ok=True)
            main_dir = data_dir / f"{sampling}" / size
            xtrain_path = main_dir / f"{i}/train_logfpkm_stdz.csv"
            ytrain_path = main_dir / f"{i}/train_labels_logfpkm.csv"
            xval_path = main_dir / f"{i}/valid_logfpkm_stdz.csv"
            yval_path = main_dir / f"{i}/valid_labels_logfpkm.csv"
            xtest_path = main_dir / f"{i}/test_logfpkm_stdz.csv"
            ytest_path = main_dir / f"{i}/test_labels_stdz.csv"

            # xtrain_path = main_dir / f"{i}/train_logrma_stdz.csv"
            # ytrain_path = main_dir / f"{i}/train_labels_logrma.csv"
            # xval_path = main_dir / f"{i}/valid_logrma_stdz.csv"
            # yval_path = main_dir / f"{i}/valid_labels_logrma.csv"
            # xtest_path = main_dir / f"{i}/test_logrma_stdz.csv"
            # ytest_path = main_dir / f"{i}/test_labels_stdz.csv"

            train_df = pd.read_csv(xtrain_path, index_col=0)
            train_labels = pd.read_csv(ytrain_path, index_col=0)

            valid_df = pd.read_csv(xval_path, index_col=0)
            valid_labels = pd.read_csv(yval_path, index_col=0)

            test_df = pd.read_csv(xtest_path, index_col=0)
            test_labels = pd.read_csv(ytest_path, index_col=0)

            y_train = train_labels[label]
            y_val = valid_labels[label]
            y_test = test_labels[label]

            assert all(y_train.index == train_df.index)
            assert all(y_val.index == valid_df.index)
            assert all(y_test.index == test_df.index)

            if any(pd.isna(y_train)):
                drop_idx = np.argwhere(pd.isna(y_train.values))
                drop_idx = drop_idx.flatten()
                train_df = train_df.drop(index=train_df.index[drop_idx])
                y_train = y_train.drop(index=y_train.index[drop_idx])

            # self.dataset = torch.tensor(self.dataset, dtype=torch.float).to(device)

            # label_embedder = LabelEncoder()
            # label_embedder.fit(y_train)
            # y_train_encoded = label_embedder.transform(y_train)
            # y_test_encoded = label_embedder.transform(y_test)

            if model == "MLP":
                train_dataset = TCGADataset(
                    train_df,
                    y_train,
                    LabelEncoder(),
                    True,
                    # sample_weights={"real": real_weight, "synthetic": synthetic_weight},
                )
                val_dataset = TCGADataset(
                    valid_df,
                    y_val,
                    train_dataset.label_embedder,
                    # sample_weights={"real": real_weight, "synthetic": synthetic_weight},
                )
                test_dataset = TCGADataset(
                    test_df, y_test, train_dataset.label_embedder
                )

                clf = CLASSIFIER_FACTORY[model](params[model], val_dataset)
                clf.fit(train_dataset, y_train)
                test_prob_preds = clf.predict_proba(test_dataset)
                test_preds = clf.predict(test_dataset)

                auc = roc_auc_score(
                    train_dataset.label_embedder.transform(y_test),
                    test_prob_preds,
                    multi_class="ovr",
                )
                bal_acc = sm.balanced_accuracy_score(
                    train_dataset.label_embedder.transform(y_test), test_preds
                )

            else:
                clf = CLASSIFIER_FACTORY[model](**params[model])
                clf.fit(train_df, y_train)
                test_prob_preds = clf.predict_proba(test_df)
                test_preds = clf.predict(test_df)
                auc = roc_auc_score(y_test, test_prob_preds, multi_class="ovr")
                bal_acc = sm.balanced_accuracy_score(y_test, test_preds)

            joblib.dump(clf, os.path.join(save_dir, f"split_{i}/{model}_model"))

            # TEST

            scores.append(auc)
            acc.append(bal_acc)

            if external:
                external_label = str.upper(label)
                external_df = pd.read_csv(
                    os.path.join(
                        data_dir, f"{sampling}", size, f"{i}/{external}_stdz.csv"
                    ),
                    index_col=0,
                )
                external_np_array = external_df.to_numpy(dtype=np.float32)
                external_labels = pd.read_csv(
                    os.path.join(
                        data_dir, f"{sampling}", size, f"{i}/{external}_labels.csv"
                    ),
                    index_col=0,
                )
                external_labels = external_labels.rename(
                    columns={external_labels.columns[0]: external_label}
                )

                external_preds = clf.predict(external_np_array)
                external_prob_preds = clf.predict_proba(external_np_array)
                # external
                drop_idx = np.argwhere(pd.isna(external_labels[external_label].values))
                keep_idx = np.setdiff1d(np.arange(len(external_preds)), drop_idx)

                external_pred_mod = external_preds[keep_idx]
                external_true_labels = external_labels[external_label][keep_idx]
                if model == "MLP":
                    external_true_labels = train_dataset.label_embedder.transform(
                        external_true_labels
                    )
                external_auc = roc_auc_score(
                    external_true_labels,
                    external_prob_preds[keep_idx],
                    multi_class="ovr",
                )
                external_bal_acc = sm.balanced_accuracy_score(
                    external_true_labels, external_pred_mod
                )
                external_scores.append(external_auc)
                external_acc.append(external_bal_acc)

        metrics["roc-auc"] = scores
        metrics["bal_acc"] = acc
        if external:
            metrics["external_roc-auc"] = external_scores
            metrics["external_bal_acc"] = external_acc
            external_mean_scores.append(np.mean(external_scores))
            external_std_scores.append(np.std(external_scores))
            external_mean_acc.append(np.mean(external_acc))
            external_std_acc.append(np.std(external_acc))

        metrics.to_csv(os.path.join(result_dir, f"{sampling}", size, "metrics.csv"))

        mean_scores.append(np.mean(scores))
        std_scores.append(np.std(scores))
        mean_acc.append(np.mean(acc))
        std_acc.append(np.std(acc))

    mean_metrics["roc-auc"] = mean_scores
    mean_metrics["std_roc-auc"] = std_scores
    mean_metrics["bal_acc"] = mean_acc
    mean_metrics["std_bal_acc"] = std_acc
    if external:
        mean_metrics["external_roc-auc"] = external_mean_scores
        mean_metrics["external_std_roc-auc"] = external_std_scores
        mean_metrics["external_bal_acc"] = external_mean_acc
        mean_metrics["external_std_bal_acc"] = external_std_acc

    mean_metrics.to_csv(
        os.path.join(result_dir, f"summary_results/mean_metrics_{size_copy}.csv")
    )


parser.add_argument(
    "data_dir", type=Path, help="Path to the main directory containing the datasets."
)
parser.add_argument("params_path", type=Path, help="Path to the JSON parameters file.")
parser.add_argument(
    "result_root_dir",
    type=Path,
    help="Path to the directory where results will be saved.",
)
parser.add_argument(
    "label", type=str, help="Name of target column to use for the classification task."
)
parser.add_argument(
    "external", type=str, help="Name of external dataset used for validation."
)
parser.add_argument("folds", type=int, default=5, help="Number of CV folds.")
parser.add_argument("repeats", type=int, default=5, help="Number of CV repeats.")
parser.add_argument(
    "seed", type=int, default=seed, help="Seed for numpy and torch, esp for mlp skorch."
)


def main(
    data_dir: str,
    params_path: str,
    result_root_dir: str,
    label: str,
    external: str,
    folds: int = 5,
    repeats: int = 5,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    sampling_methods = [
        "poisson",
        "unmod_poisson",
        "unmod_gamma_poisson",
        "gamma_poisson",
        "local_crossover",
        "global_crossover",
        "smote",
        "replacement",
        "unaugmented",
    ]

    for model in ["Logistic", "KNN", "RF", "EBM", "SVM-RBF", "MLP"]:
        for size in ["max", "50", "500"]:
            run_clf(
                model,
                data_dir,
                result_root_dir,
                sampling_methods,
                params_path,
                size,
                label,
                external,
                folds,
                repeats,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.data_dir,
        args.params_path,
        args.result_root_dir,
        args.label,
        args.external,
        args.folds,
        args.repeats,
        args.seed,
    )
