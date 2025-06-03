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
@click.option("--signature_path", type=click.Path(path_type=Path, exists=True))
def main(data_dir, external_name, external_xpath, external_ypath, signature_path):
    signature_genes = pd.read_csv(signature_path)

    sampling_methods = [
        f for f in data_dir.iterdir() if f.is_dir() and "data" not in f.stem
    ]

    ext_df = pd.read_csv(external_xpath, index_col=0)
    if "ENS" in ext_df.columns[1]:
        ext_df = ext_df.rename(
            columns=dict(zip(signature_genes["ensemblid"], signature_genes["SYMBOL"]))
        )
    ext_labels = pd.read_csv(external_ypath, index_col=0)

    ext_df = ext_df.loc[ext_labels.index, :]

    for method in sampling_methods:
        if "unaugmented" in method.stem:
            size = [""]
        else:
            size = ["max", "500", "5000"]
        for s in size:
            subfolders = list((method / s).rglob("**/scaler.pkl"))
            for folder in subfolders[1:]:
                scaler = joblib.load(folder)
                stdz_external_dataset(
                    scaler, external_name, ext_df, ext_labels, folder.parent
                )


if __name__ == "__main__":
    main()
