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

warnings.filterwarnings("ignore")

root_dir = Path("/Users/nja/Desktop/Sinergia")

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
    def fpkm(df: pd.DataFrame, lengths: pd.DataFrame, patient_counts: np.array):

        fpkm_df = df.apply(lambda x: (x * 10**9) / lengths, axis=1)
        assert fpkm_df.iloc[0, 1] == df.iloc[0, 1] * 10**9 / lengths[1]
        fpkm_df = fpkm_df.apply(lambda x: x / patient_counts, axis=0)

        return fpkm_df

    # revert from log to count
    # count_df = count_df.applymap(lambda x: 2**x - 1)
    # get length of genes present inn the df
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
# @click.option(
#     "--cv_splits_path",
#     required=True,
#     type=click.Path(path_type=Path, exists=True),
#     help="Path to json file containing cross-validation splits wrt indices.",
# )
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
# @click.option(
#     "--validation_size",
#     type=float,
#     help="Size of validation set when splitting train set for hyperparam tuning.",
# )
@click.option("--seed", type=int, help="Seed for train-test split function.")
@click.option(
    "--save_dir",
    type=click.Path(path_type=Path),
    help="Directory in which to save the augmented data.",
)
def main(
    # cv_splits_path: Path,
    ref_df_path: Path,
    ref_labels_path: Path,
    # probemap_path: Path,
    # signature_path: Path,
    sampling_method: str,
    class_size: int,
    target: str,
    # validation_size: float,
    seed: int,
    save_dir: Path,
):

    save_dir.mkdir(parents=True, exist_ok=True)
    class_size = literal_eval(class_size)

    ref_df = pd.read_csv(ref_df_path, index_col=0)
    ref_labels = pd.read_csv(ref_labels_path, index_col=0)

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

    gene_lengths = probemap["length"][sampled_df.columns]
    sampled_df = get_fpkm_from_count(sampled_df, gene_lengths, standardise=False)

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
    test_df_stdz = pd.DataFrame(
        scaler.transform(test_df),
        columns=test_df.columns,
        index=test_df.index,
    )
    # test_fpkm_stdz = (test_fpkm_df - training_mean) / training_std
    test_df_stdz.to_csv(os.path.join(save_dir, "real_test_df_stdz.csv"))
    test_labels.to_csv(os.path.join(save_dir, "real_test_labels.csv"))


if __name__ == "__main__":
    main()
