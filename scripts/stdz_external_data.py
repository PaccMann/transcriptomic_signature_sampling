from pathlib import Path

import click
import joblib
import pandas as pd

from signature_sampling.utils import stdz_external_dataset


@click.command()
@click.option("--data_dir", type=click.Path(path_type=Path, exists=True))
@click.option("--external_name", type=str)
@click.option("--external_xpath", type=click.Path(path_type=Path, exists=True))
@click.option("--external_ypath", type=click.Path(path_type=Path, exists=True))
@click.option("--colotype_path", type=click.Path(path_type=Path, exists=True))
def main(data_dir, external_name, external_xpath, external_ypath, colotype_path):
    colotype_genes = pd.read_csv(colotype_path)

    sampling_methods = [
        f for f in data_dir.iterdir() if f.is_dir() and "data" not in f.stem
    ]

    cptac_df = pd.read_csv(external_xpath, index_col=0)
    cptac_df = cptac_df.rename(
        columns=dict(zip(colotype_genes["ensemblid"], colotype_genes["SYMBOL"]))
    )
    cptac_labels = pd.read_csv(external_ypath, index_col=0)

    for method in sampling_methods:
        if "unaugmented" in method.stem:
            size = [""]
        else:
            size = ["max", "500", "5000"]
        for s in size:
            subfolders = list((method / s).rglob("**/"))
            for folder in subfolders[1:]:
                scaler = joblib.load(folder / "scaler.pkl")
                stdz_external_dataset(
                    scaler, external_name, cptac_df, cptac_labels, folder
                )


if __name__ == "__main__":
    main()
