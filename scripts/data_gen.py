import json
import os
import warnings
from ast import literal_eval
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from signature_sampling.hyperparameter_factory import SAMPLING_FACTORY
from signature_sampling.utils import fpkm, fpkm_normalised_df

warnings.filterwarnings("ignore")


@click.command()
@click.option(
    "--cv_splits_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Path to json file containing cross-validation splits wrt indices.",
)
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
    "--probemap_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Path to probemap file that maps genes to ensembl ids and specifies gene lengths.",
)
@click.option(
    "--signature_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Path to file containing info about signature genes and their subtypes.",
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
    "--validation_size",
    type=float,
    help="Size of validation set when splitting train set for hyperparam tuning.",
)
@click.option("--seed", type=int, help="Seed for train-test split function.")
@click.option(
    "--save_dir",
    type=click.Path(path_type=Path),
    help="Directory in which to save the augmented data.",
)
def main(
    cv_splits_path: Path,
    ref_df_path: Path,
    ref_labels_path: Path,
    probemap_path: Path,
    signature_path: Path,
    sampling_method: str,
    class_size: int,
    target: str,
    validation_size: float,
    seed: int,
    save_dir: Path,
):
    # size = "max" if class_size is None else class_size

    # save_path = os.path.join(save_dir, f"{sampling_method}", f"{size}") in shell script
    # os.makedirs(save_path, exist_ok=True)
    if class_size == "max":
        class_size = None
    else:
        class_size = literal_eval(class_size)
    probemap = pd.read_csv(probemap_path, sep="\t")
    probemap.index = [i.split(".")[0] for i in probemap.id]
    probemap = probemap.drop(columns="id")
    probemap["length"] = probemap.chromEnd - probemap.chromStart
    # Need to fix gene name
    probemap.loc["ENSG00000140105", "gene"] = "WARS1"
    probemap.set_index("gene", inplace=True)

    with open(cv_splits_path, "r") as f:
        cv_splits = json.load(f)

    ref_df = pd.read_csv(ref_df_path, index_col=0)
    ref_labels = pd.read_csv(ref_labels_path, index_col=0)
    assert all(ref_df.index == ref_labels.index)
    # ref_df = ref_df.applymap(lambda x: 2**x - 1)  # undo the log transformation
    # remove ensembl version ID if present
    if any("ENS" in x for x in ref_df.columns):
        ref_df.columns = [i.split(".")[0] for i in ref_df.columns]
        gene_type = "ensemblid"
    else:
        gene_type = "SYMBOL"

    # TODO: add primary tumour check

    if os.path.exists(colotype_path):
        colotype_genes = pd.read_csv(colotype_path)
    else:
        colotype_genes = dict()

    signature_genes = dict(
        colotype_genes.groupby("subtype").apply(lambda x: x[gene_type].values)
    )

    splits = len(cv_splits.keys())

    for i in range(splits):
        train_idx = cv_splits[str(i)]["train"]
        train_df = ref_df.loc[train_idx, :]
        train_labels = pd.DataFrame(ref_labels.loc[train_idx, :], columns=[target])
        train_df.index.name = None
        train_labels.index.name = None
        # load and process test
        test_idx = cv_splits[str(i)]["test"]
        test_df = ref_df.loc[test_idx, :]
        test_labels = pd.DataFrame(ref_labels.loc[test_idx, :], columns=[target])
        test_df.index.name = None
        test_labels.index.name = None

        # sampler_obj = Sampler()
        if sampling_method != "unaugmented":
            save_path = save_dir / f"{i+1}"
            sampler = SAMPLING_FACTORY[sampling_method](sampling_method, class_size)
            sampler.init_target_signatures(signature_genes)
            sampled_df, sampled_labels = sampler.sample(train_df, train_labels, target, **{'gamma_poisson_r':5, 'poisson_r':5, 'overlapping_genes':['MCM2','MKI67']})
            augmented_df = pd.concat([train_df, sampled_df])
            augmented_labels = pd.concat([train_labels, sampled_labels])
            shuffle_idx = np.random.permutation(range(len(augmented_df)))

            # real+synthetic samples of counts and labels
            augmented_df = augmented_df.iloc[shuffle_idx]
            augmented_labels = augmented_labels.iloc[shuffle_idx]

        else:
            augmented_df = train_df
            augmented_labels = train_labels
            save_path = save_dir.parent / f"{i+1}"

        # save augmented count data
        os.makedirs(save_path, exist_ok=True)
        augmented_df.to_csv(os.path.join(save_path, "train_counts_colotype.csv"))
        augmented_labels.to_csv(os.path.join(save_path, "train_labels_colotype.csv"))

        # FPKM normalise
        augmented_fpkm_df = fpkm_normalised_df(probemap, augmented_df)

        Xtrain, Xval, ytrain, yval = train_test_split(
            augmented_fpkm_df,
            augmented_labels,
            test_size=validation_size,
            random_state=seed,
            stratify=augmented_labels,
        )

        scaler = StandardScaler()
        train_augmented_fpkm_df_stdz = pd.DataFrame(
            scaler.fit_transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index
        )
        valid_augmented_fpkm_df_stdz = pd.DataFrame(
            scaler.transform(Xval), columns=Xval.columns, index=Xval.index
        )
        joblib.dump(scaler, save_path / "scaler.pkl")

        # save augmented logfpkm data and stdz params
        train_augmented_fpkm_df_stdz.to_csv(
            save_path / "train_logfpkm_stdz.csv"
        )
        valid_augmented_fpkm_df_stdz.to_csv(
            save_path / "valid_logfpkm_stdz.csv"
        )
        ytrain.to_csv(os.path.join(save_path, "train_labels_logfpkm.csv"))
        yval.to_csv(os.path.join(save_path, "valid_labels_logfpkm.csv"))

        # standardise test set based on training set
        assert all(train_augmented_fpkm_df_stdz.columns == test_df.columns)
        test_fpkm_df = fpkm_normalised_df(probemap, test_df)
        test_fpkm_stdz = pd.DataFrame(
            scaler.transform(test_fpkm_df),
            columns=test_fpkm_df.columns,
            index=test_fpkm_df.index,
        )
        # test_fpkm_stdz = (test_fpkm_df - training_mean) / training_std
        test_fpkm_stdz.to_csv(os.path.join(save_path, "test_logfpkm_stdz.csv"))
        test_labels.to_csv(os.path.join(save_path, "test_labels_stdz.csv"))


if __name__ == "__main__":
    main()
