import argparse
import json
import os
import warnings
from ast import literal_eval
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from signature_sampling.hyperparameter_factory import CLASSIFIER_FACTORY
from signature_sampling.torch_data import TCGADataset

warnings.filterwarnings("ignore")
seed = 42
parser = argparse.ArgumentParser()
np.random.seed(seed)
torch.manual_seed(seed)


def run_clf(
    models: List[str],
    data_dir: Path,
    result_dir: Path,
    params_path: Path,
    label: str = "target",
    external: str = "real_test",
    **kwargs,
) -> None:
    """Saves results from classification experiment in csv format in the designated folder.

    Args:
        model (List[str]): List of Classification model to use. Refer hyperparameter factory for options.
        data_dir (str): Path to the data folder.
        result_root_dir (str): Path to where the results should be saved.
        label (str, optional): The clinical label to use for the classification task.
            Defaults to "cms".
        folds (int, optional): Number of cross validation folds. Defaults to 5.
        repeats (int, optional): Number of cross validation repetitions. Defaults to 5.
        external (bool, optional): Whether to run the experiment on external data. Defaults to True.
    """

    with open(params_path, "r") as f:
        params = json.load(f)

    mean_scores = []
    mean_cm = []
    mean_acc = []
    mean_f1 = []

    external_mean_scores = []
    external_mean_cm = []
    external_mean_acc = []
    external_mean_f1 = []
    confusion_matrix_dict = {}

    mean_metrics = pd.DataFrame(
        columns=["roc-auc", "bal_acc", "external_roc-auc", "external_bal_acc"],
    )

    xtrain_path = data_dir / "train_stdz.csv"
    ytrain_path = data_dir / "train_labels.csv"
    xval_path = data_dir / "val_stdz.csv"
    yval_path = data_dir / "val_labels.csv"
    xtest_path = data_dir / "test_stdz.csv"
    ytest_path = data_dir / "test_labels.csv"

    train_df = pd.read_csv(xtrain_path, index_col=0).iloc[:, :40]
    assert train_df.shape[1] == 40
    train_labels = pd.read_csv(ytrain_path, index_col=0)

    valid_df = pd.read_csv(xval_path, index_col=0).iloc[:, :40]
    valid_labels = pd.read_csv(yval_path, index_col=0)

    test_df = pd.read_csv(xtest_path, index_col=0).iloc[:, :40]
    test_labels = pd.read_csv(ytest_path, index_col=0)

    y_train = train_labels[label]
    y_val = valid_labels[label]
    y_test = test_labels[label]

    assert all(y_train.index == train_df.index)
    assert all(y_val.index == valid_df.index)
    assert all(y_test.index == test_df.index)

    for model in models:
        if "probability" in params[model].keys():
            params[model]["probability"] = literal_eval(params[model]["probability"])

        if model == "MLP":
            train_dataset = TCGADataset(
                train_df,
                y_train,
                LabelEncoder(),
                True,
            )
            val_dataset = TCGADataset(
                valid_df,
                y_val,
                train_dataset.label_embedder,
            )
            test_dataset = TCGADataset(test_df, y_test, train_dataset.label_embedder)

            clf = CLASSIFIER_FACTORY[model](params[model], val_dataset)
            clf.fit(train_dataset, y_train)
            test_prob_preds = clf.predict_proba(test_dataset)
            test_preds = clf.predict(test_dataset)
            true_test_labels = train_dataset.label_embedder.transform(y_test)

            auc = roc_auc_score(
                true_test_labels,
                test_prob_preds,
                multi_class="ovr",
            )
            bal_acc = sm.balanced_accuracy_score(true_test_labels, test_preds)
            confusion_matrix_array = confusion_matrix(true_test_labels, test_preds)
            f1 = f1_score(true_test_labels, test_preds, average="micro")

        else:
            clf = CLASSIFIER_FACTORY[model](**params[model])
            clf.fit(train_df, y_train)
            test_prob_preds = clf.predict_proba(test_df)
            test_preds = clf.predict(test_df)
            auc = roc_auc_score(y_test, test_prob_preds, multi_class="ovr")
            bal_acc = sm.balanced_accuracy_score(y_test, test_preds)
            confusion_matrix_array = confusion_matrix(y_test, test_preds)
            f1 = f1_score(y_test, test_preds, average="micro")

        # TEST

        mean_scores.append(auc)
        mean_cm.append(confusion_matrix_array)
        mean_acc.append(bal_acc)
        mean_f1.append(f1)

        if external:
            external_label = "cms"
            external_df = pd.read_csv(
                os.path.join(data_dir, f"{external}_df_stdz.csv"),
                index_col=0,
            ).iloc[:, :40]
            external_df.rename(
                columns=dict(zip(external_df.columns, range(external_df.shape[1]))),
                inplace=True,
            )

            external_labels = pd.read_csv(
                os.path.join(data_dir, f"{external}_labels.csv"),
                index_col=0,
            )
            label_enc = LabelEncoder()
            external_true_labels = label_enc.fit_transform(
                external_labels[external_label]
            )

            if model == "MLP":

                ext_dataset = TCGADataset(
                    external_df,
                    external_true_labels,
                    train_dataset.label_embedder,
                )

                external_preds = clf.predict(ext_dataset)
                external_prob_preds = clf.predict_proba(ext_dataset)

            else:

                external_preds = clf.predict(external_df)
                external_prob_preds = clf.predict_proba(external_df)

            external_auc = roc_auc_score(
                external_true_labels,
                external_prob_preds,
                multi_class="ovr",
            )
            external_bal_acc = sm.balanced_accuracy_score(
                external_true_labels, external_preds
            )
            external_confusion_matrix_array = confusion_matrix(
                external_true_labels, external_preds
            )
            external_f1 = f1_score(
                external_true_labels, external_preds, average="micro"
            )

            external_mean_scores.append(external_auc)
            external_mean_cm.append(external_confusion_matrix_array)
            external_mean_acc.append(external_bal_acc)
            external_mean_f1.append(external_f1)

            mean_metrics.loc[model, "roc-auc"] = auc
            mean_metrics.loc[model, "bal_acc"] = bal_acc
            confusion_matrix_dict[model] = {
                "confusion_matrix": confusion_matrix_array.tolist()
            }
            mean_metrics.loc[model, "f1"] = f1
            if external:
                mean_metrics.loc[model, "external_roc-auc"] = external_auc
                mean_metrics.loc[model, "external_bal_acc"] = external_bal_acc
                confusion_matrix_dict[model] = {
                    "external_confusion_matrix": external_confusion_matrix_array.tolist()
                }
                mean_metrics.loc[model, "external_f1"] = external_f1

    return mean_metrics, confusion_matrix_dict


parser.add_argument(
    "data_root_dir",
    type=Path,
    help="Path to the main directory containing the datasets.",
)
parser.add_argument("params_path", type=Path, help="Path to the JSON parameters file.")
parser.add_argument(
    "result_dir",
    type=Path,
    help="Path to the directory where results will be saved.",
)
parser.add_argument(
    "label", type=str, help="Name of target column to use for the classification task."
)
parser.add_argument(
    "external", type=str, help="Name of external dataset used for validation."
)
parser.add_argument(
    "seed", type=int, default=seed, help="Seed for numpy and torch, esp for mlp skorch."
)


def main(
    data_root_dir: Path,
    params_path: str,
    result_dir: Path,
    label: str,
    external: str,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    models = ["RF", "KNN", "MLP"]

    result_df_list = []
    confusion_matrix_final = {}
    for data_dir in data_root_dir.iterdir():
        if data_dir.name == ".DS_Store":
            continue
        seed_split = data_dir.name
        mean_metrics_split, confusion_matrix_dict = run_clf(
            models,
            data_dir,
            result_dir,
            params_path,
            label,
            external,
        )
        mean_metrics_split["seed_split"] = [seed_split] * mean_metrics_split.shape[0]
        result_df_list.append(mean_metrics_split)
        confusion_matrix_final[seed_split] = confusion_matrix_dict

    pd.concat(result_df_list).to_csv(
        os.path.join(result_dir, "baseline_metrics_repeated.csv")
    )
    with open(os.path.join(result_dir, "baseline_cm_repeated.csv"), "w") as f:
        json.dump(confusion_matrix_final, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.data_root_dir,
        args.params_path,
        args.result_dir,
        args.label,
        args.external,
        args.seed,
    )
