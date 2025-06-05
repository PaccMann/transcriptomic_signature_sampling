import os
import warnings
from ast import literal_eval
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from signature_sampling.hyperparameter_factory import SAMPLING_FACTORY
from signature_sampling.utils import fpkm

warnings.filterwarnings("ignore")

root_dir = Path("")

probemap = pd.read_csv(
    root_dir / "data/gdc_pancan/gene_expression/gencode.v22.annotation.gene.probeMap",
    sep="\t",
)
# probemap.index = probemap.id
# probemap = probemap.drop(columns='id')
probemap["length"] = probemap.chromEnd - probemap.chromStart
probemap = probemap.replace({"WARS": "WARS1"})
probemap = probemap.set_index("gene")


def get_fpkm_from_count(
    count_df: pd.DataFrame, lengths: pd.DataFrame, standardise: bool
):
    gene_lengths = lengths[count_df.columns]
    assert all(gene_lengths.index == count_df.columns)
    # get patient wise sum of counts
    patient_sum = np.sum(count_df, axis=1)
    assert len(patient_sum) == len(count_df)
    assert np.allclose(patient_sum[0], sum(count_df.iloc[0, :]))
    # get fpkm df and log2 transform
    fpkm_df = fpkm(count_df, gene_lengths, patient_sum).applymap(
        lambda x: np.log2(x + 1)
    )

    if standardise == True:
        scaler = StandardScaler()
        fpkm_stdz = pd.DataFrame(
            scaler.fit_transform(fpkm_df), columns=fpkm_df.columns, index=fpkm_df.index
        )
        return fpkm_stdz

    return fpkm_df


@click.command()
@click.option(
    "--ref_df_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Path to dataset to be split and augmented in csv format.",
)
@click.option(
    "--ref_labels_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Path to labels associated with the dataset in csv format.",
)
@click.option(
    "--sampling_method",
    required=True,
    type=click.Choice(sorted(SAMPLING_FACTORY.keys())),
    help="Sampling method to use for augmentation.",
)
@click.option("--class_size", type=str, help="Total desired size for each class.")
@click.option("--target", type=str, help="Target variable name.")
@click.option(
    "--save_dir",
    type=click.Path(path_type=Path),
    help="Directory in which to save the augmented data.",
)
@click.option(
    "--genes_to_postprocess",
    required=False,
    type=click.Path(path_type=Path, exists=True),
    help="Path to genes to subset and postprocess; columns name should be 'SYMBOL'.",
)
def main(
    ref_df_path: Path,
    ref_labels_path: Path,
    sampling_method: str,
    class_size: int,
    target: str,
    save_dir: Path,
    genes_to_postprocess: Path,
):

    save_dir.mkdir(parents=True, exist_ok=True)
    class_size = literal_eval(class_size)

    ref_df = pd.read_csv(ref_df_path, index_col=0)
    ref_labels = pd.read_csv(ref_labels_path, index_col=0)

    genes = pd.read_csv(genes_to_postprocess, index_col=0)
    genes = genes["SYMBOL"]

    # load and process test

    test_df = ref_df
    test_labels = ref_labels
    sampler = SAMPLING_FACTORY[sampling_method](sampling_method, class_size)

    sampled_df, sampled_labels = sampler.sample(
        ref_df,
        ref_labels,
        target,
        **{
            "gamma_poisson_r": 5,
            "poisson_r": 5,
        },
    )

    shuffle_idx = np.random.permutation(range(len(sampled_df)))

    sampled_df = sampled_df.iloc[shuffle_idx]
    sampled_labels = sampled_labels.iloc[shuffle_idx]

    assert test_df.merge(sampled_df, "inner").empty

    # apply post-processing to genes

    gene_lengths = probemap["length"][sampled_df[genes].columns]
    sampled_df[genes] = get_fpkm_from_count(
        sampled_df[genes], gene_lengths, standardise=False
    )

    scaler = StandardScaler()
    augmented_train_df_stdz = pd.DataFrame(
        scaler.fit_transform(sampled_df),
        columns=sampled_df.columns,
        index=sampled_df.index,
    )

    joblib.dump(scaler, save_dir / "scaler.pkl")

    # save augmented data
    augmented_train_df_stdz.to_csv(save_dir / "augmented_train_df_stdz.csv")
    sampled_labels.to_csv(os.path.join(save_dir, "augmented_train_labels.csv"))

    # standardise test set based on training set
    assert all(augmented_train_df_stdz.columns == test_df.columns)
    # apply post-processing to genes

    gene_lengths = probemap["length"][test_df[genes].columns]
    test_df[genes] = get_fpkm_from_count(
        test_df[genes], gene_lengths, standardise=False
    )
    test_df_stdz = pd.DataFrame(
        scaler.transform(test_df),
        columns=test_df.columns,
        index=test_df.index,
    )

    test_df_stdz.to_csv(os.path.join(save_dir, "real_test_df_stdz.csv"))
    test_labels.to_csv(os.path.join(save_dir, "real_test_labels.csv"))


if __name__ == "__main__":
    main()
