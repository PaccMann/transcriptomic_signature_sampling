import json
import os
from pathlib import Path
from typing import Iterable

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

seed = 42
np.random.seed(seed)


def eval_clf(
    model: Path,
    data_dir: Path,
    result_dir: Path,
    label: str = "cms",
    external: str = "cptac",
    **kwargs,
) -> None:

    ## load model
    with open(result_dir / f"{model}_model", "rb") as f:
        clf = joblib.load(f)

    ## process test data to retrieve labels
    external_label = str.upper(label)
    external_df = pd.read_csv(
        os.path.join(data_dir / f"{external}_stdz.csv"),
        index_col=0,
    )
    external_np_array = external_df.to_numpy(dtype=np.float32)
    external_labels = pd.read_csv(
        os.path.join(data_dir / f"{external}_labels.csv"),
        index_col=0,
    )
    external_labels = external_labels.rename(
        columns={external_labels.columns[0]: external_label}
    )

    external_preds = clf.predict(external_np_array)

    # external
    drop_idx = np.argwhere(pd.isna(external_labels[external_label].values))
    keep_idx = np.setdiff1d(np.arange(len(external_preds)), drop_idx)

    external_pred_mod = external_preds[keep_idx]
    external_true_labels = external_labels[external_label].iloc[keep_idx]

    np.save(result_dir / f"{external}_preds.npy", external_pred_mod)

    cm_norm = confusion_matrix(
        external_true_labels,
        external_pred_mod,
        labels=["LumA", "LumB", "Basal"],
        normalize="true",
    )

    return cm_norm.flatten()


def main(
    data_dir: Path = Path("/data/brca_microarray/GSE_augmentations"),
    result_root_dir: Path = Path("/results/brca_microarray_results"),
    data="",
    filename="confusion_matrices_brca.csv",
):
    """
    Save multiple confusion matrices to a CSV file with metadata.
    """
    # Create column names for confusion matrix entries
    cm_columns = []
    for true_class in ["LumA", "LumB", "Basal"]:
        for pred_class in ["LumA", "LumB", "Basal"]:
            cm_columns.append(f"True {true_class} - Pred {pred_class}")

    # Create metadata column names
    meta_columns = ["Sampling Method", "Model", "Class-Size", "Split"]

    # Create empty DataFrame with all columns
    all_columns = meta_columns + cm_columns
    df = pd.DataFrame(columns=all_columns)

    # Process each confusion matrix
    sampling_methods = [
        "poisson",
        # "unmod_poisson",
        # "unmod_gamma_poisson",
        "gamma_poisson",
        "local_crossover",
        "global_crossover",
        "smote",
        "replacement",
        "unaugmented",
    ]
    i = 0
    for sampling_method in sampling_methods:
        for model in ["Logistic", "KNN", "RF", "EBM", "SVM-RBF"]:
            for size in ["max", "50", "500"]:
                for split in range(1, 26):
                    print(
                        f"Computing confusion matrices for {sampling_method, model, size, split}"
                    )

                    if sampling_method == "unaugmented":
                        result_dir = (
                            result_root_dir
                            / data
                            / model
                            / sampling_method
                            / f"split_{split}"
                        )
                        data_folder = data_dir / data / sampling_method / f"{split}"
                    else:
                        result_dir = (
                            result_root_dir
                            / data
                            / model
                            / sampling_method
                            / size
                            / f"split_{split}"
                        )
                        data_folder = (
                            data_dir / data / sampling_method / size / f"{split}"
                        )
                    cm_norm_flattened = eval_clf(
                        model,
                        data_folder,
                        result_dir,
                        label="Pam50 + Claudin-low subtype",
                        external="metabric",
                    )

                    # Create a row with confusion matrix values and metadata
                    row_data = np.concatenate(
                        [
                            [sampling_method, model, size, split],
                            cm_norm_flattened,
                        ]
                    )

                    # Add row to DataFrame
                    df.loc[i] = row_data
                    i += 1

    # Save DataFrame to CSV
    df.to_csv(result_root_dir / filename, index=False)


if __name__ == "__main__":
    main()
