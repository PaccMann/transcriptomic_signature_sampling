import argparse
import json
import os

import numpy as np
import pandas as pd
from signature_sampling.sampler import Sampler
from signature_sampling.utils import fpkm
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument(
    "params_path",
    type=str,
    help="Path to the JSON parameter file specifying hyperparameters.",
)
parser.add_argument(
    "probemap_path",
    type=str,
    help="Path to the probemap file containing information about gene lengths. Found in the data link provided.",
)
parser.add_argument(
    "save_path", type=str, help="Path where the data files should be saved."
)


def main(
    params_path: str,
    probemap_path: str,
    save_dir: str,
):
    with open(params_path, "r") as f:
        params = json.load(f)

    sampling = params.get("sampling", "smote")
    target = params.get("target", "cms")
    class_size = eval(params.get("class_size", "None"))
    size = "max" if class_size is None else class_size
    valid_size = params.get("valid_size", 0.1)
    seed = params.get("split_seed", 9)

    save_path = os.path.join(save_dir, f"{sampling}", f"{size}")
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "params.json"), "w") as f:
        json.dump(params, f)

    probemap = pd.read_csv(probemap_path, sep="\t")
    probemap.index = [i.split(".")[0] for i in probemap.id]
    probemap = probemap.drop(columns="id")
    probemap["length"] = probemap.chromEnd - probemap.chromStart

    ref_df = pd.read_csv(params["reference_data"], index_col=0)
    ref_df = ref_df.applymap(lambda x: 2**x - 1)  # undo the log transformation

    test_data = pd.read_csv(params["test_data"], index_col=0)
    test_data = test_data.applymap(lambda x: 2**x - 1)

    # remove ensembl version ID if present
    if any("ENS" in x for x in ref_df.columns):
        ref_df.columns = [i.split(".")[0] for i in ref_df.columns]
        test_data.columns = [i.split(".")[0] for i in test_data.columns]
        gene_type = "ensemblid"
    else:
        gene_type = "SYMBOL"

    ref_labels = pd.read_csv(params["reference_labels"], index_col=0)
    assert all(ref_df.index == ref_labels.index)

    if os.path.exists(params.get("colotype_path", "")):
        colotype_genes = pd.read_csv(params["colotype_path"])
    else:
        colotype_genes = dict()

    signature_genes = dict(
        colotype_genes.groupby("subtype").apply(lambda x: x[gene_type].values)
    )
    kwargs = {"target_signatures": signature_genes}

    sampler_obj = Sampler()

    if sampling == "smote":
        sampling_strategy = params.get("sampling_strategy", "auto")

        augmented_df, augmented_labels = sampler_obj.smote_sampler(
            ref_df, ref_labels, target, sampling_strategy
        )

    elif "crossover_global" in sampling:
        sampled_df, sampled_labels = sampler_obj.crossover_sampler_global(
            ref_df, ref_labels, target, class_size, signature_genes
        )

        augmented_df = pd.concat([ref_df, sampled_df])
        augmented_labels = pd.concat([ref_labels, sampled_labels])

    else:
        sampled_df, sampled_labels = sampler_obj.subset_sampler(
            ref_df,
            ref_labels,
            target,
            class_size,
            sampling,
            **kwargs,
        )

        # sampled_df = sampled_df.loc[:, ref_df.columns]

        augmented_df = pd.concat([ref_df, sampled_df])
        augmented_labels = pd.concat([ref_labels, sampled_labels])

    shuffle_idx = np.random.permutation(range(len(augmented_df)))

    # real+synthetic samples of counts and labels
    augmented_df = augmented_df.iloc[shuffle_idx]
    augmented_labels = augmented_labels.iloc[shuffle_idx]

    # save augmented count data
    augmented_df.to_csv(os.path.join(save_path, "train_counts_colotype.csv"))
    augmented_labels.to_csv(os.path.join(save_path, "train_labels_colotype.csv"))

    # FPKM normalise
    gene_lengths = probemap["length"][augmented_df.columns]
    assert all(gene_lengths.index == augmented_df.columns)
    # get patient wise sum of counts
    patient_sum = np.sum(augmented_df, axis=1)
    assert len(patient_sum) == len(augmented_df)
    assert np.allclose(patient_sum[0], sum(augmented_df.iloc[0, :]))
    # get fpkm df and log2 transform
    augmented_fpkm_df = fpkm(augmented_df, gene_lengths, patient_sum).applymap(
        lambda x: np.log2(x + 1)
    )

    Xtrain, Xval, ytrain, yval = train_test_split(
        augmented_fpkm_df,
        augmented_labels,
        test_size=valid_size,
        random_state=split_seed,
        stratify=augmented_labels,
    )

    training_mean = Xtrain.mean()
    training_std = Xtrain.std()
    train_augmented_fpkm_df_stdz = (Xtrain - training_mean) / training_std
    valid_augmented_fpkm_df_stdz = (Xval - training_mean) / training_std
    stdz_params = pd.DataFrame({"mean": training_mean, "std": training_mean})

    # save augmented logfpkm data and stdz params
    stdz_params.to_csv(os.path.join(save_path, "train_stdz_params.csv"))
    train_augmented_fpkm_df_stdz.to_csv(
        os.path.join(save_path, "train_logfpkm_colotype_stdz.csv")
    )
    valid_augmented_fpkm_df_stdz.to_csv(
        os.path.join(save_path, "valid_logfpkm_colotype_stdz.csv")
    )
    ytrain.to_csv(os.path.join(save_path, "train_labels_logfpkm_colotype.csv"))
    yval.to_csv(os.path.join(save_path, "valid_labels_logfpkm_colotype.csv"))

    # standardise test set based on training set
    assert all(train_augmented_fpkm_df_stdz.columns == test_data.columns)
    # test_data.to_csv(os.path.join(save_path,"test_counts_colotype.csv"))
    # FPKM normalise
    gene_lengths = probemap["length"][test_data.columns]
    assert all(gene_lengths.index == test_data.columns)
    # get patient wise sum of counts
    patient_sum = np.sum(test_data, axis=1)
    assert len(patient_sum) == len(test_data)
    assert np.allclose(patient_sum[0], sum(test_data.iloc[0, :]))
    # get fpkm df and log2 transform
    test_fpkm_df = fpkm(test_data, gene_lengths, patient_sum).applymap(
        lambda x: np.log2(x + 1)
    )
    test_fpkm_stdz = (test_fpkm_df - training_mean) / training_std
    test_fpkm_stdz.to_csv(os.path.join(save_path, "test_logfpkm_colotype_stdz.csv"))
    test_labels = pd.read_csv(params["test_labels"], index_col=0)
    test_labels.to_csv(os.path.join(save_path, "test_labels_colotype_stdz.csv"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.params_path,
        args.probemap_path,
        args.save_path,
    )
