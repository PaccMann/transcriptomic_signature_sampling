import json
import os
import random
from pathlib import Path
from random import sample
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import torch
from scipy.stats import wilcoxon
from sklearn import metrics

from signature_sampling.vae import VAE


def fpkm(
    df: pd.DataFrame, lengths: pd.Series, patient_counts: pd.Series
) -> pd.DataFrame:
    """Returns FPKM normalised Dataframe.

    Args:
        df (pd.DataFrame): Count dataframe to be FPKM normalised.
        lengths (pd.Series): Lengths of genes, where genes are indices.
        patient_counts (pd.Series): Total patient wise gene count (Row sum).
        Indices are patients.

    Returns:
        pd.DataFrame: FPKM normalised RNA-Seq dataframe.
    """

    fpkm_df = df.apply(lambda x: (x * 10**9) / lengths, axis=1)
    assert fpkm_df.iloc[0, 1] == df.iloc[0, 1] * 10**9 / lengths[1]
    fpkm_df = fpkm_df.apply(lambda x: x / patient_counts, axis=0)

    return fpkm_df


def fpkm_normalised_df(probemap, df):
    gene_lengths = probemap["length"][df.columns]
    assert all(gene_lengths.index == df.columns)
    # get patient wise sum of counts
    patient_sum = np.sum(df, axis=1)
    assert len(patient_sum) == len(df)
    assert np.allclose(patient_sum[0], sum(df.iloc[0, :]))
    # get fpkm df and log2 transform
    fpkm_df = fpkm(df, gene_lengths, patient_sum).applymap(lambda x: np.log2(x + 1))

    return fpkm_df


def purity_score(y_true, y_pred):
    """Computes purity score of predicted clusters.
        Source: https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
    Args:
        y_true(np.ndarray): Column vector of true target labels.
        y_pred(np.ndarray): Column vector of predicted cluster assignments.

    Returns:
        float: Purity score
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def significance_testing(
    result_dir: str,
    sampling_methods: Iterable,
    train_sizes: Dict = {"max": 622, "500": 1800, "5000": 18000},
    cptac: bool = True,
    filename: str = "metrics.csv",
) -> Tuple:
    """Computes p-value using Wilcoxon test and saves it in csv format.

    Args:
        result_dir (str): Path where the cluster purity scores across all splits are saved.
        sampling_methods (Iterable): List of sampling methods to compute significance.
        train_sizes (_type_, optional): Dictionary summarising the total training data
            available for each class size. Defaults to {"max": 622, "500": 1800, "5000": 18000}.
        cptac (bool, optional): Whether to compute significance for cptac as well. Defaults to True.

    Returns:
        Tuple: Dataframes of the wilcoxon scores comparing different sampling methods and
            their associated perfromances on the tcga and cptac data.
    """
    sizes = train_sizes.keys()
    for size in sizes:
        w_df = pd.DataFrame(columns=sampling_methods, index=sampling_methods)

        cptac_w_df = pd.DataFrame(columns=sampling_methods, index=sampling_methods)

        for sampling in sampling_methods:
            if "unbalanced" in sampling:
                size_sample = ""
            else:
                size_sample = size
            for comparison in sampling_methods:
                if "unbalanced" in comparison:
                    size_comp = ""
                else:
                    size_comp = size

                if sampling == comparison:
                    continue

                results_path_sample = os.path.join(
                    result_dir, sampling, size_sample, filename
                )

                results_path_comp = os.path.join(
                    result_dir, comparison, size_comp, filename
                )

                results_sample = pd.read_csv(results_path_sample)
                results_comp = pd.read_csv(results_path_comp)

                w = wilcoxon(
                    results_sample["0"], results_comp["0"], alternative="greater"
                )

                w_df.loc[sampling, comparison] = w

                if cptac:
                    results_path_sample = os.path.join(
                        result_dir, sampling, size_sample, f"cptac_{filename}"
                    )

                    results_path_comp = os.path.join(
                        result_dir, comparison, size_comp, f"cptac_{filename}"
                    )

                    w_cptac = wilcoxon(
                        results_path_sample["0"],
                        results_path_comp["0"],
                        alternative="greater",
                    )

                    cptac_w_df.loc[sampling, comparison] = w_cptac

    return w_df, cptac_w_df


def load_model(model_name: str, model_path: str):
    saved_model_params_path = os.path.join(model_path, "params.json")
    saved_model = os.path.join(model_path, "weights")

    with open(saved_model_params_path, "r") as readjson:
        saved_model_params = json.load(readjson)

    model = VAE(saved_model_params)

    state_dict_enc = torch.load(
        os.path.join(saved_model, model_name), map_location=torch.device("cpu")
    )["model_state_dict"]
    for key in list(state_dict_enc.keys()):
        state_dict_enc[key.replace("0.0.", "0.")] = state_dict_enc.pop(key)

    model.load_state_dict(state_dict_enc)

    return model


def get_latent_embeddings(
    model: Callable,
    input_data: torch.Tensor,
    results_dir: str,
    filename: str,
) -> torch.Tensor:
    """_summary_

    Args:
        model (Callable): _description_
        input_data (torch.Tensor): _description_
        results_dir (str): _description_
        filename (str): _description_

    Returns:
        torch.Tensor: _description_
    """
    os.makedirs(results_dir, exist_ok=True)

    model.eval()

    z_mean, z_logvar = model.encoder(input_data)
    latent_embed, q_z, p_z = model.reparameterise(z_mean, z_logvar)

    torch.save(latent_embed, os.path.join(results_dir, filename))

    return latent_embed


def get_decoded_embeddings(
    model: Callable,
    latent_embedding: torch.Tensor,
) -> torch.tensor:
    model.eval()
    decoded_embedding = model.decoder(latent_embedding)
    reconstructed_embedding = model.final_layer(decoded_embedding)

    return reconstructed_embedding


def subset_fraction(cv_splits: dict, percent: float = 0.1, seed: int = 42) -> dict:
    """_summary_

    Args:
        cv_splits (dict): _description_
        percent (float): _description_
        seed (int): _description_

    Returns:
        dict: _description_
    """
    np.random.seed(seed)
    random.seed(seed)

    split = sample(list(cv_splits.keys()), 1)[0]
    length = len(cv_splits[split]["train"])
    subset_size = math.floor(percent * length)

    for i in cv_splits.keys():
        cv_splits[i]["train"] = sample(cv_splits[i]["train"], subset_size)

    return cv_splits


def stdz_external_dataset(
    scaler: sklearn.preprocessing,
    external_name: str,
    external_df: pd.DataFrame,
    external_labels: pd.DataFrame,
    save_dir: Path,
):
    external_df = external_df.loc[:, scaler.feature_names_in_]
    assert all(external_df.columns == scaler.feature_names_in_)
    external_stdz = pd.DataFrame(
        scaler.transform(external_df),
        index=external_df.index,
        columns=external_df.columns,
    )
    assert all(external_stdz.index == external_labels.index)

    external_stdz.to_csv(save_dir / f"{external_name}_stdz.csv")
    external_labels.to_csv(save_dir / f"{external_name}_labels.csv")


def save_dict(dictionary: dict, save_path: str) -> None:
    with open(save_path, "w") as f:
        json.dump(dictionary, f)
