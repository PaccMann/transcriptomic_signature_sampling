import os
from pathlib import PurePath
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from cms_classifier.utils.clustering import Clustering


from signature_sampling.clustering import Clustering
from signature_sampling.utils import purity_score
from typing import Iterable, Callable, Tuple


def get_best_k(
    result_root_dir: str,
    method: str,
    size: str,
    n_clusters: Iterable = [2, 3, 4, 5, 6, 7, 8, 9, 10],
) -> Callable:
    """Returns the KMeans model with the optimal number of clusters.

    Args:
        result_root_dir (str): Root directory of the results.
        method (str): Sampling method.
        size (str): Class size to which the data is augmented by the sampling method.
        n_clusters (Iterable, optional): _description_. Defaults to [2, 3, 4, 5, 6, 7, 8, 9, 10].

    Returns:
        Callable: Best KMeans model across the provided range of number of clusters.
    """
    result_main_dir = PurePath(result_root_dir) / method / size / "split_1" / "results"

    train_latent = np.vstack(
        torch.load(
            os.path.join(result_main_dir, "best_latent_train"), map_location="cpu"
        )
    )
    val_latent = np.vstack(
        torch.load(os.path.join(result_main_dir, "best_latent_val"), map_location="cpu")
    )

    merged_data = pd.concat([pd.DataFrame(train_latent), pd.DataFrame(val_latent)])

    best_km = Clustering().kmeans_elbow(merged_data, n_clusters)
    print(f"best_k for {method}, {size} = ", best_km.n_clusters)
    return best_km


def bestk_cluster_purity(
    result_root_dir: str,
    data_root_dir: str,
    method: str,
    size: str,
    splits: int,
    best_k: int,
) -> Tuple:
    """Computes the cluster purity of the training and test data using the best KMeans model.

    Args:
        result_root_dir (str): Root directory of the results.
        data_root_dir (str): Root directory of the data.
        method (str): Sampling method.
        size (str): Class size to which the data is augmented by the sampling method.
        splits (int): Total number of train/test splits. For ex, splits = 25 for a 5x5 CV.
        best_k (int): The best number of clusters as determined by the elbow method.

    Returns:
        Tuple: Tuple of the train (TCGA), test (TCGA) and external test (CPTAC) cluster purities across all splits.
    """
    train_purities = []
    test_purities = []
    cptac_purities = []
    for split in range(1, splits + 1):
        result_main_dir = (
            PurePath(result_root_dir) / method / size / f"split_{split}" / "results"
        )
        result_save_dir = PurePath(result_root_dir) / method / size
        data_main_dir = PurePath(data_root_dir) / method / size / str(split)

        train_latent = np.vstack(
            torch.load(
                os.path.join(result_main_dir, "best_latent_train"), map_location="cpu"
            )
        )
        val_latent = np.vstack(
            torch.load(
                os.path.join(result_main_dir, "best_latent_val"), map_location="cpu"
            )
        )
        train_labels = np.hstack(
            torch.load(
                os.path.join(result_main_dir, "best_train_labels"), map_location="cpu"
            )
        )
        val_labels = np.hstack(
            torch.load(
                os.path.join(result_main_dir, "best_val_labels"), map_location="cpu"
            )
        )

        merged_data = pd.concat([pd.DataFrame(train_latent), pd.DataFrame(val_latent)])
        merged_labels = pd.concat(
            [pd.DataFrame(train_labels), pd.DataFrame(val_labels)]
        )

        clustering_obj, clustering_labels, cluster_centres = Clustering().clustering(
            merged_data, "kmeans", best_k
        )

        train_purity = purity_score(merged_labels, clustering_labels)

        train_purities.append(train_purity)

        test_latent = np.vstack(
            torch.load(os.path.join(result_main_dir, "test_latent"), map_location="cpu")
        )
        test_labels = pd.read_csv(
            os.path.join(data_main_dir, "test_labels_logfpkm_colotype.csv"), index_col=0
        )

        test_pred = clustering_obj.predict(test_latent)
        test_purity = purity_score(test_labels, test_pred)
        test_purities.append(test_purity)

        cptac_latent = np.vstack(
            torch.load(
                os.path.join(result_main_dir, "cptac_latent"), map_location="cpu"
            ).detach()
        )
        cptac_labels = pd.read_csv(
            os.path.join(data_main_dir, "cptac_labels.csv"), index_col=0
        )
        cptac_labels = LabelEncoder().fit_transform(cptac_labels)
        cptac_labels_mod = cptac_labels[cptac_labels != 4]
        drop_idx = np.argwhere(cptac_labels == 4)
        keep_idx = np.setdiff1d(np.arange(len(cptac_labels)), drop_idx)

        cptac_latent = cptac_latent[keep_idx, :]

        cptac_pred = clustering_obj.predict(cptac_latent)
        cptac_purity = purity_score(cptac_labels_mod, cptac_pred)
        cptac_purities.append(cptac_purity)

    print(
        method,
        size,
        "train_purity = ",
        np.mean(train_purities),
        "±",
        np.std(train_purities),
    )
    print(
        method,
        size,
        "test_purity = ",
        np.mean(test_purities),
        "±",
        np.std(test_purities),
    )
    print(
        method,
        size,
        "cptac_purity = ",
        np.mean(cptac_purities),
        "±",
        np.std(cptac_purities),
    )
    pd.DataFrame(train_purities).to_csv(
        os.path.join(result_save_dir, "kmeans_purity_train_splits.csv")
    )
    pd.DataFrame(test_purities).to_csv(
        os.path.join(result_save_dir, "kmeans_purity_test_splits.csv")
    )
    pd.DataFrame(cptac_purities).to_csv(
        os.path.join(result_save_dir, "kmeans_purity_cptac_splits.csv")
    )
    return train_purities, test_purities, cptac_purities
