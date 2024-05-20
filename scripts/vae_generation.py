# TODO - refactor !!! put gmm code in a func and save gmm model
import json
import os
import warnings
from ast import literal_eval
from pathlib import Path
from typing import Counter, Dict, Iterable

# from collections import Counter
import click
import joblib
import numpy as np
import pandas as pd
import torch
from numpy import random
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from torch.distributions.normal import Normal

from signature_sampling.hyperparameter_factory import CLASSIFIER_FACTORY
from signature_sampling.utils import (get_decoded_embeddings,
                                      get_latent_embeddings, load_model)

warnings.filterwarnings("ignore")


def generate_classify(
    clf_model: str,
    params: dict,
    data_dir: str,
    model_dir: str,
    vae_model_name: str,
    result_root_dir: str,
    sampling_methods: Iterable,
    size: str,
    cv_splits_idx: Dict,
    external: bool = False,
    n_splits=25,
    seed=42,
) -> None:
    """Function that runs a generation-prediction experiment and saves results
        in the designated folder.

    Args:
        clf_model (str): Trained classification model to use for the downstream task.
        clf_params (dict): Dictionary of parameters to initialise the classifier model.
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
    result_dir = os.path.join(result_root_dir, f"VAE_{clf_model}")
    os.makedirs(os.path.join(result_dir, "summary_results"), exist_ok=True)
    mean_acc = []
    std_acc = []
    mean_auc = []
    std_auc = []
    cptac_mean_acc = []
    cptac_std = []
    cptac_mean_auc = []
    cptac_std_auc = []
    size_copy = size
    mean_metrics = pd.DataFrame(
        columns=["bal_acc", "std_bal_acc", "cptac_bal_acc", "std_cptac_bal_acc"],
        index=sampling_methods,
    )

    for sampling in sampling_methods:
        scores = []
        acc = []
        cptac_acc = []
        cptac_scores = []

        metrics = pd.DataFrame(columns=["bal_acc", "cptac_bal_acc"])

        if sampling == "unaugmented":
            size = ""

        save_dir = os.path.join(result_dir, f"{sampling}", size)

        for split in range(1, n_splits + 1):
            os.makedirs(os.path.join(save_dir, f"split_{split}"), exist_ok=True)

            data_path = os.path.join(data_dir, sampling, size, f"{split}")

            test_data = pd.read_csv(
                os.path.join(data_path, "test_logfpkm_colotype_stdz.csv"), index_col=0
            )

            test_labels = pd.read_csv(
                os.path.join(data_path, "test_labels_colotype_stdz.csv"), index_col=0
            )

            test_ids = cv_splits_idx[str(split - 1)]["test"]

            test_data = test_data.loc[test_ids, :]
            test_target = test_labels.loc[test_ids, :]

            assert all(test_data.index == test_target.index)

            test_tensor = torch.from_numpy(np.asarray(test_data, dtype=np.float32))

            test_weights = dict(sorted(Counter(test_labels.values.ravel()).items()))

            if external:
                external_test_data = pd.read_csv(
                    os.path.join(data_path, "cptac_stdz.csv"), index_col=0
                )

                # TODO
                external_target = pd.read_csv(
                    os.path.join(data_path, "cptac_labels.csv"), index_col=0
                )
                drop_idx = external_target.index[
                    pd.isna(external_target.values.ravel())
                ]

                external_test_data = external_test_data.drop(index=drop_idx)
                external_target = external_target.drop(index=drop_idx)
                assert all(external_test_data.index == external_target.index)

                external_test_tensor = torch.from_numpy(
                    np.asarray(external_test_data, dtype=np.float32)
                )
                ext_weights = dict(
                    sorted(Counter(external_target.values.ravel()).items())
                )
            # train_ids = cv_splits_idx[str(split - 1)]["train"]

            model_path = os.path.join(
                model_dir, vae_model_name, sampling, size, f"run_{split}"
            )
            results_path = os.path.join(result_dir, sampling, size, f"split_{split}")

            model = load_model(vae_model_name, model_path)

            test_latent = get_latent_embeddings(
                model, test_tensor, results_path, "test_latent"
            )

            test_latent = pd.DataFrame(test_latent.detach().numpy())

            test_latent_labels = pd.DataFrame(
                np.concatenate([test_latent, test_target], axis=-1),
                index=test_data.index,
                columns=list(range(test_latent.shape[-1])) + ["true_labels"],
            )

            grouped_df_mean = test_latent_labels.groupby("true_labels").agg("mean")
            grouped_df_std = test_latent_labels.groupby("true_labels").agg("std")

            generated_test_samples = pd.DataFrame()
            generated_test_labels = pd.DataFrame()
            generated_samples = []
            generated_labels = []
            for i, label in enumerate(grouped_df_mean.index):
                mean = torch.from_numpy(np.float32(grouped_df_mean.iloc[i, :].values))
                std = torch.from_numpy(np.float32(grouped_df_std.iloc[i, :].values))
                class_gaussian = Normal(mean, std)
                generated_class = class_gaussian.sample_n(test_weights[label])
                generated_samples.append(generated_class)
                generated_labels.append([label] * len(generated_class))

            generated_test_tensor = torch.concatenate(generated_samples)
            generated_test_samples = get_decoded_embeddings(
                model, generated_test_tensor
            )

            generated_test_samples = pd.DataFrame(
                generated_test_samples.detach().numpy()
            )
            generated_test_embed = pd.DataFrame(generated_test_tensor.detach().numpy())
            generated_test_labels = pd.DataFrame(np.concatenate(generated_labels))

            generated_test_samples.to_csv(
                os.path.join(results_path, "vae_generated_test_samples.csv")
            )
            generated_test_embed.to_csv(
                os.path.join(results_path, "vae_generated_test_latent.csv")
            )
            generated_test_labels.to_csv(
                os.path.join(results_path, "vae_generated_test_labels.csv")
            )

            if external:
                # TODO - get external labels
                external_latent = get_latent_embeddings(
                    model,
                    external_test_tensor,
                    results_path,
                    "cptac_latent",
                )

                external_latent = external_latent.detach().numpy()
                columns = list(range(external_latent.shape[-1])) + [
                    "true_labels",
                ]
                external_latent_labels = pd.DataFrame(
                    np.concatenate([external_latent, external_target], axis=-1),
                    index=external_test_data.index,
                    columns=columns,
                )

                ext_grouped_df_mean = external_latent_labels.groupby("true_labels").agg(
                    "mean"
                )
                ext_grouped_df_std = external_latent_labels.groupby("true_labels").agg(
                    "std"
                )

                generated_samples = []
                generated_labels = []
                for i, label in enumerate(ext_grouped_df_mean.index):
                    mean = torch.from_numpy(
                        np.float32(ext_grouped_df_mean.iloc[i, :].values)
                    )
                    std = torch.from_numpy(
                        np.float32(ext_grouped_df_std.iloc[i, :].values)
                    )
                    class_gaussian = Normal(mean, std)
                    generated_class = class_gaussian.sample_n(ext_weights[label])
                    generated_samples.append(generated_class)
                    generated_labels.append([label] * len(generated_class))

                generated_ext_tensor = torch.concatenate(generated_samples)
                generated_ext_samples = get_decoded_embeddings(
                    model, generated_ext_tensor
                )

                generated_ext_samples = pd.DataFrame(
                    generated_ext_samples.detach().numpy()
                )
                generated_ext_embed = pd.DataFrame(
                    generated_ext_tensor.detach().numpy()
                )
                generated_ext_labels = pd.DataFrame(np.concatenate(generated_labels))

                generated_ext_samples.to_csv(
                    os.path.join(results_path, "vae_generated_ext_samples.csv")
                )
                generated_ext_embed.to_csv(
                    os.path.join(results_path, "vae_generated_ext_latent.csv")
                )
                generated_ext_labels.to_csv(
                    os.path.join(results_path, "vae_generated_ext_labels.csv")
                )

            clf_model_path = os.path.join(
                result_root_dir,
                clf_model,
                sampling,
                size,
                f"split_{split}/{clf_model}_model",
            )
            classifier = joblib.load(clf_model_path)

            test_preds = classifier.predict(generated_test_samples)
            test_preds_prob = classifier.predict_proba(generated_test_samples)

            # TEST

            bal_acc = balanced_accuracy_score(generated_test_labels, test_preds)

            acc.append(bal_acc)

            auc = roc_auc_score(
                generated_test_labels[0],
                test_preds_prob,
                multi_class="ovr",
            )
            scores.append(auc)

            if external:
                cptac_preds = classifier.predict(generated_ext_samples)
                cptac_prob = classifier.predict_proba(generated_ext_samples)

                cptac_bal_acc = balanced_accuracy_score(
                    generated_ext_labels, cptac_preds
                )
                cptac_acc.append(cptac_bal_acc)
                cptac_auc = roc_auc_score(
                    generated_ext_labels[0],
                    cptac_prob,
                    multi_class="ovr",
                )
                cptac_scores.append(cptac_auc)

        metrics["bal_acc"] = acc
        metrics["auc_roc"] = scores
        if external:
            metrics["cptac_bal_acc"] = cptac_acc
            cptac_mean_acc.append(np.mean(cptac_acc))
            cptac_std.append(np.std(cptac_acc))
            cptac_mean_auc.append(np.mean(cptac_auc))
            cptac_std_auc.append(np.std(cptac_auc))

        metrics.to_csv(os.path.join(result_dir, f"{sampling}", size, "metrics.csv"))

        mean_acc.append(np.mean(acc))
        std_acc.append(np.std(acc))
        mean_auc.append(np.mean(auc))
        std_auc.append(np.std(auc))

    mean_metrics["bal_acc"] = mean_acc
    mean_metrics["std_bal_acc"] = std_acc
    mean_metrics["roc_auc"] = mean_auc
    mean_metrics["std_roc_auc"] = std_auc

    if external:
        mean_metrics["cptac_bal_acc"] = cptac_mean_acc
        mean_metrics["std_cptac_bal_acc"] = cptac_std
        mean_metrics["cptac_roc_auc"] = cptac_mean_auc
        mean_metrics["std_cptac_roc_auc"] = cptac_std_auc

    mean_metrics.to_csv(
        os.path.join(result_dir, f"summary_results/mean_metrics_{size_copy}.csv")
    )


@click.command()
@click.option(
    "--clf_model",
    type=click.Choice(sorted(CLASSIFIER_FACTORY.keys())),
    help="Classification model to use.",
)
@click.option(
    "--params_path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to generator-classifier parameters.",
)
@click.option("--size", type=str, help="class size of dataset.")
@click.option(
    "--data_dir",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Root directory of datasets.",
)
@click.option(
    "--model_dir",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Directory where the VAE model is saved.",
)
@click.option(
    "--result_root_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory where results should be saved.",
)
@click.option(
    "--cv_splits_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Path to the CV splits json file.",
)
@click.option("--external", type=bool, help="Whether to test an external dataset.")
@click.option("--seed", type=int, help="Seed for reproducibility.")
def main(
    clf_model,
    params_path,
    size,
    data_dir,
    model_dir,
    result_root_dir,
    cv_splits_path,
    external,
    seed,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    vae_model_name = "NVAE"

    with open(cv_splits_path, "r") as f:
        cv_splits_idx = json.load(f)

    with open(params_path, "r") as f:
        params = json.load(f)

    sampling_methods = [
        "poisson",
        "gamma_poisson",
        "local_crossover",
        "global_crossover",
        "smote",
        "replacement",
        "unaugmented",
    ]

    generate_classify(
        clf_model,
        params,
        data_dir,
        model_dir,
        vae_model_name,
        result_root_dir,
        sampling_methods,
        size,
        cv_splits_idx,
        external,
        n_splits=25,
    )


if __name__ == "__main__":
    main()
