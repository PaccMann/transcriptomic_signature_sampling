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

parser = argparse.ArgumentParser()


def run_clf(
    model: str,
    data_dir: Path,
    result_root_dir: Path,
    sampling_methods: Iterable,
    params_path: Path,
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

    with open(params_path, "r") as f:
        params = json.load(f)
    
    if "probability" in params[model].keys():
        params[model]["probability"] = literal_eval(params[model]["probability"])

    mean_scores = []
    std_scores = []
    mean_acc = []
    std_acc = []
    cptac_mean_scores = []
    cptac_std_scores = []
    cptac_mean_acc = []
    cptac_std_acc = []
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

        if sampling == "unaugmented":
            size = ""

        save_dir = os.path.join(result_dir, f"{sampling}", size)

        for i in range(1, (folds * repeats) + 1):
            os.makedirs(os.path.join(save_dir, f"split_{i}"), exist_ok=True)
            main_dir = data_dir / f"{sampling}" / size
            xtrain_path = main_dir / f"{i}/train_logfpkm_colotype_stdz.csv"
            ytrain_path = main_dir / f"{i}/train_labels_logfpkm_colotype.csv"
            xval_path = main_dir / f"{i}/valid_logfpkm_colotype_stdz.csv"
            yval_path = main_dir / f"{i}/valid_labels_logfpkm_colotype.csv"
            xtest_path = main_dir / f"{i}/test_logfpkm_colotype_stdz.csv"
            ytest_path = main_dir / f"{i}/test_labels_colotype_stdz.csv"

            train_df = pd.read_csv(xtrain_path, index_col=0)
            train_labels = pd.read_csv(ytrain_path, index_col=0)

            valid_df = pd.read_csv(xval_path, index_col=0)
            valid_labels = pd.read_csv(yval_path, index_col=0)

            test_df = pd.read_csv(xtest_path, index_col=0)
            test_labels = pd.read_csv(ytest_path, index_col=0)

            y_train = train_labels[label]
            y_val = valid_labels[label]
            y_test = test_labels[label]

            if any(pd.isna(y_train)):
                drop_idx = np.argwhere(pd.isna(y_train.values))
                drop_idx = drop_idx.flatten()
                train_df = train_df.drop(index=train_df.index[drop_idx])
                y_train = y_train.drop(index=y_train.index[drop_idx])

            # single param json, with params_LR, etc for each model
            # do **params_LR to init the clf and make it ready for fit
            # TODO: convert data to numpy and labels to int with labelencoder

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

                cptac_preds = clf.predict(cptac_df.to_numpy(dtype=np.float32))
                cptac_prob_preds = clf.predict_proba(
                    cptac_df.to_numpy(dtype=np.float32)
                )
                # CPTAC
                drop_idx = np.argwhere(pd.isna(cptac_labels[cptac_label].values))
                keep_idx = np.setdiff1d(np.arange(len(cptac_preds)), drop_idx)

                cptac_pred_mod = cptac_preds[keep_idx]
                cptac_true_labels = cptac_labels[cptac_label][keep_idx]
                if model == "MLP":
                    cptac_true_labels = train_dataset.label_embedder.transform(
                        cptac_true_labels
                    )
                cptac_auc = roc_auc_score(
                    cptac_true_labels,
                    cptac_prob_preds[keep_idx],
                    multi_class="ovr",
                )
                cptac_bal_acc = sm.balanced_accuracy_score(
                    cptac_true_labels, cptac_pred_mod
                )
                cptac_scores.append(cptac_auc)
                cptac_acc.append(cptac_bal_acc)

        metrics["roc-auc"] = scores
        metrics["bal_acc"] = acc
        if cptac:
            metrics["cptac_roc-auc"] = cptac_scores
            metrics["cptac_bal_acc"] = cptac_acc
            cptac_mean_scores.append(np.mean(cptac_scores))
            cptac_std_scores.append(np.std(cptac_scores))
            cptac_mean_acc.append(np.mean(cptac_acc))
            cptac_std_acc.append(np.std(cptac_acc))

        metrics.to_csv(os.path.join(result_dir, f"{sampling}", size, "metrics.csv"))

        mean_scores.append(np.mean(scores))
        std_scores.append(np.std(scores))
        mean_acc.append(np.mean(acc))
        std_acc.append(np.std(acc))

    mean_metrics["roc-auc"] = mean_scores
    mean_metrics["std_roc-auc"] = std_scores
    mean_metrics["bal_acc"] = mean_acc
    mean_metrics["std_bal_acc"] = std_acc
    if cptac:
        mean_metrics["cptac_roc-auc"] = cptac_mean_scores
        mean_metrics["cptac_std_roc-auc"] = cptac_std_scores
        mean_metrics["cptac_bal_acc"] = cptac_mean_acc
        mean_metrics["cptac_std_bal_acc"] = cptac_std_acc

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
parser.add_argument("seed", type=int, help="Seed for numpy and torch, esp for mlp skorch.")

def main(data_dir: str, params_path: str, result_root_dir: str, seed:int):

    np.random.seed(seed)
    torch.manual_seed(seed)

    sampling_methods = [
        "poisson",
        "gamma_poisson",
        "local_crossover",
        "global_crossover",
        "smote",
        "replacement",
        "unaugmented"
    ]
    for model in ["MLP","RF","Logistic","SVM-RBF", "KNN", "EBM"]:
        for size in ["max", "500", "5000"]:
            run_clf(
                model, data_dir, result_root_dir, sampling_methods, params_path, size
            )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.data_dir, args.params_path, args.result_root_dir, args.seed)
