import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import joblib
from typing import Counter, Iterable, Dict
from signature_sampling.utils import load_model, get_latent_embeddings
from signature_sampling.hyperparameter_factory import CLASSIFIER_FACTORY


def predict_clinical(
    clf_model: str,
    data_dir: str,
    model_dir: str,
    vae_model_name: str,
    result_root_dir: str,
    sampling_methods: Iterable,
    size: str,
    cv_splits_idx: Dict,
    clinical_data: pd.DataFrame,
    clinical_var: str,
    external: bool = False,
    external_clinical=None,
    n_splits=25,
) -> None:
    """Function that runs a clinical variable prediction experiment and saves results
        in the designated folder.

    Args:
        clf_model (str): Classification model to use for the downstream task.
        data_dir (str): Root directory of the datasets.
        model_dir (str): Path to the saved vae model.
        vae_model_name (str): Name of the saved vae model.
        result_root_dir (str): Path where the generated results are saved.
        sampling_methods (Iterable): List of augmentation methods to compare.
        size (str): Class size to which augmentation is performed.
        cv_splits_idx (Dict): Nested dictionary containing the split number as key and the train
            and test indices associated with that split as the values.
        clinical_data (pd.DataFrame): Dataframe containing the clinical data indexed by
            patient ID.
        clinical_var (str): Name of clinical variable as it appears in the dataframe to
            be used for the prediction task.
        external (bool, optional): Whether to run the experiment on an external dataset.
          Defaults to True.
        external_clinical (_type_, optional): Dataframe of clinical data associated with
            the external data. Defaults to None.
        n_splits (int, optional): Number of cross validation splits as measured
            by repeats * folds. Defaults to 25 (5 fold repeated 5 times).
    """
    result_dir = os.path.join(result_root_dir, clf_model)
    os.makedirs(os.path.join(result_dir, "summary_results"), exist_ok=True)
    mean_acc = []
    std_acc = []
    cptac_mean_acc = []
    cptac_std = []
    size_copy = size
    mean_metrics = pd.DataFrame(
        columns=["bal_acc", "cptac_bal_acc"],
        index=sampling_methods,
    )
    std_metrics = pd.DataFrame(
        columns=["bal_acc", "cptac_bal_acc"], index=sampling_methods
    )

    for sampling in sampling_methods:
        scores = []
        acc = []

        cptac_acc = []
        metrics = pd.DataFrame(columns=["bal_acc", "cptac_bal_acc"])

        if sampling == "unbalanced":
            size = ""

        save_dir = os.path.join(result_dir, f"{sampling}", size)

        scores = []
        acc = []

        metrics = pd.DataFrame(columns=["bal_acc"])

        save_dir = os.path.join(result_dir, f"{sampling}", size)

        for split in range(1, n_splits + 1):
            os.makedirs(os.path.join(save_dir, f"split_{split}"), exist_ok=True)

            data_path = os.path.join(data_dir, sampling, size, f"{split}")

            training_data = pd.read_csv(
                os.path.join(data_path, "train_logfpkm_colotype_stdz.csv"), index_col=0
            )
            validation_data = pd.read_csv(
                os.path.join(data_path, "valid_logfpkm_colotype_stdz.csv"), index_col=0
            )
            train_data = pd.concat([training_data, validation_data])
            test_data = pd.read_csv(
                os.path.join(data_path, "test_logfpkm_colotype_stdz.csv"), index_col=0
            )

            if external:
                external_test_data = pd.read_csv(
                    os.path.join(data_path, "cptac_stdz.csv"), index_col=0
                )
                external_merged_data = external_test_data.join(
                    external_clinical[clinical_var]
                ).dropna()
                external_features = external_merged_data.drop(columns=clinical_var)

                external_test_tensor = torch.from_numpy(
                    np.asarray(external_features, dtype=np.float32)
                )
                external_target = pd.DataFrame(
                    external_merged_data[clinical_var], columns=[clinical_var]
                )
                assert all(external_features.index == external_target.index)

            train_ids = cv_splits_idx[str(split - 1)]["train"]
            test_ids = cv_splits_idx[str(split - 1)]["test"]

            real_train_data = train_data.loc[train_ids, :]
            test_data = test_data.loc[test_ids, :]

            train_clinical = clinical_data[clinical_data.index.isin(train_ids)]
            test_clinical = clinical_data[clinical_data.index.isin(test_ids)]

            train_merged_data = real_train_data.join(train_clinical[clinical_var])
            test_merged_data = test_data.join(test_clinical[clinical_var])

            train_merged_data = train_merged_data.dropna()
            test_merged_data = test_merged_data.dropna()

            train_features = train_merged_data.drop(columns=clinical_var)
            train_target = train_merged_data[clinical_var]
            test_features = test_merged_data.drop(columns=clinical_var)
            test_target = test_merged_data[clinical_var]

            assert all(train_features.index == train_target.index)
            assert all(test_features.index == test_target.index)

            if split == 1 and sampling == "unbalanced":
                print("train len", len(train_target), Counter(train_target))
                print("test len", len(test_target), Counter(test_target))
                if external:
                    print(
                        "cptac len",
                        len(external_target),
                        Counter(external_target[clinical_var]),
                    )

            train_tensor = torch.from_numpy(
                np.asarray(train_features, dtype=np.float32)
            )
            test_tensor = torch.from_numpy(np.asarray(test_features, dtype=np.float32))

            model_path = os.path.join(
                model_dir, vae_model_name, sampling, size, f"split_{split}"
            )
            results_path = os.path.join(result_dir, sampling, size, f"split_{split}")

            model = load_model(vae_model_name, model_path)

            train_latent = get_latent_embeddings(
                model, vae_model_name, train_tensor, results_path, "train_latent"
            )
            test_latent = get_latent_embeddings(
                model, vae_model_name, test_tensor, results_path, "test_latent"
            )
            if external:
                external_latent = get_latent_embeddings(
                    model,
                    vae_model_name,
                    external_test_tensor,
                    results_path,
                    "cptac_latent",
                )

            estimator = CLASSIFIER_FACTORY[clf_model]
            estimator.fit(train_latent.detach().numpy(), train_target)
            joblib.dump(
                estimator, os.path.join(save_dir, f"split_{split}/{clf_model}_model")
            )

            # TEST

            bal_acc = balanced_accuracy_score(
                test_target, estimator.predict(test_latent.detach().numpy())
            )

            acc.append(bal_acc)

            if external:
                cptac_preds = estimator.predict(external_latent.detach().numpy())

                # CPTAC
                drop_idx = np.argwhere(pd.isna(external_target[clinical_var].values))
                keep_idx = np.setdiff1d(np.arange(len(cptac_preds)), drop_idx)

                cptac_pred_mod = cptac_preds[keep_idx]
                cptac_bal_acc = balanced_accuracy_score(
                    external_target[clinical_var][keep_idx], cptac_pred_mod
                )
                cptac_acc.append(cptac_bal_acc)

        metrics["bal_acc"] = acc
        if external:
            metrics["cptac_bal_acc"] = cptac_acc
            cptac_mean_acc.append(np.mean(cptac_acc))
            cptac_std.append(np.std(cptac_acc))

        metrics.to_csv(os.path.join(result_dir, f"{sampling}", size, "metrics.csv"))

        mean_acc.append(np.mean(acc))
        std_acc.append(np.std(acc))

    mean_metrics["bal_acc"] = mean_acc
    std_metrics["bal_acc"] = std_acc
    if external:
        mean_metrics["cptac_bal_acc"] = cptac_mean_acc
        std_metrics["cptac_bal_acc"] = cptac_std

    mean_metrics.to_csv(
        os.path.join(result_dir, f"summary_results/mean_metrics_{size_copy}.csv")
    )
    std_metrics.to_csv(
        os.path.join(result_dir, f"summary_results/std_metrics_{size_copy}.csv")
    )
