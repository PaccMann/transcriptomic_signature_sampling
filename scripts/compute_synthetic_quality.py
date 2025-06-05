import os
import re
from pathlib import Path

import joblib
import pandas as pd
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
from sdmetrics.reports.single_table._properties.coverage import Coverage
from sdmetrics.single_table import LogisticDetection
from sklearn.metrics import silhouette_score

SYN_PATTERN = re.compile("S[0-9]")
TRAIN_FILENAME = "train_logfpkm_colotype_stdz.csv"
TRAIN_FILENAME_LABELS = "train_labels_logfpkm_colotype.csv"
VAL_FILENAME = "valid_logfpkm_colotype_stdz.csv"
VAL_FILENAME_LABELS = "valid_labels_logfpkm_colotype.csv"
SCALER = "scaler.pkl"

# TRAIN_FILENAME = "train_logrma_stdz.csv"
# TRAIN_FILENAME_LABELS = "train_labels_logrma.csv"
# VAL_FILENAME = "valid_logrma_stdz.csv"
# VAL_FILENAME_LABELS = "valid_labels_logrma.csv"


def get_metadata(table: pd.DataFrame):

    metadata = {}
    table.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
    columns = table.columns.to_list()
    columns_type = {"ID": {"sdtype": "id", "regex_format": ""}}
    columns_type.update(
        {
            col: {"sdtype": "numerical", "compute_representation": "Float"}
            for col in columns[1:]
        }
    )
    metadata["primary_key"] = "ID"
    metadata["columns"] = columns_type

    return metadata, table


def process_train_val_data(
    data_path,
    train_fname,
    train_label_fname,
    val_fname,
    val_label_fname,
    target="cms",
):
    xtrain = pd.read_csv(data_path / train_fname)
    ytrain = pd.read_csv(data_path / train_label_fname, index_col=0)
    xval = pd.read_csv(data_path / val_fname)
    yval = pd.read_csv(data_path / val_label_fname, index_col=0)

    with open(data_path / SCALER, "rb") as f:
        scaler = joblib.load(f)

    data = pd.concat([xtrain, xval])
    labels = pd.concat([ytrain, yval])
    metadata, data = get_metadata(data)

    data_unstdz = pd.DataFrame(
        scaler.inverse_transform(data.drop(columns="ID")),
        index=data["ID"],
        columns=data.columns[1:],
    )
    data.set_index("ID", inplace=True, drop=False)
    syn_ids = data.index.str.contains(SYN_PATTERN)
    data = pd.concat([data["ID"], data_unstdz], axis=1)

    syn_data = data.loc[syn_ids, :]
    real_data = data.loc[~syn_ids, :]
    syn_labels = labels.loc[syn_ids, target]
    real_labels = labels.loc[~syn_ids, target]

    return syn_data, syn_labels, real_data, real_labels, metadata


def compute_synthetic_quality(data_path: Path, save_dir: Path):

    syn_data, syn_labels, real_data, real_labels, metadata = process_train_val_data(
        data_path,
        TRAIN_FILENAME,
        TRAIN_FILENAME_LABELS,
        VAL_FILENAME,
        VAL_FILENAME_LABELS,
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    qual_report = QualityReport()
    qual_report.generate(real_data, syn_data, metadata)

    diag_report = DiagnosticReport()
    diag_report.generate(real_data, syn_data, metadata)

    detection_score = LogisticDetection.compute(real_data, syn_data, metadata)
    print("detection_score", detection_score)

    cov = Coverage()
    coverage = cov.get_score(real_data, syn_data, metadata)

    quality = qual_report.get_properties()
    diag = diag_report.get_properties()

    Shape = quality["Score"][0]
    Trend = quality["Score"][1]
    Validity = diag["Score"][0]
    Structure = diag["Score"][1]

    syn_sil_score = silhouette_score(
        syn_data.drop(columns="ID"), syn_labels, metric="cosine"
    )
    real_sil_score = silhouette_score(
        real_data.drop(columns="ID"), real_labels, metric="cosine"
    )

    with open(f"{save_dir}/quality_unstdz.txt", "w") as f:
        f.write(f"Shape: {Shape}\n")
        f.write(f"Trend: {Trend}\n")
        f.write(f"Validity: {Validity}\n")
        f.write(f"Structure: {Structure}\n")
        f.write(f"Detection: {detection_score}\n")
        f.write(f"Coverage: {coverage}\n")
        f.write(f"synthetic_silhouette_score: {syn_sil_score}\n")
        f.write(f"real_silhouette_score: {real_sil_score}\n")

    shapes = qual_report.get_details(property_name="Column Shapes")
    trends = qual_report.get_details(property_name="Column Pair Trends")

    shapes.to_csv(f"{save_dir}/shape_unstdz.csv")
    trends.to_csv(f"{save_dir}/trend_unstdz.csv")


CLASS_SIZE = 5000

for method in [
    "gamma_poisson",
    "global_crossover",
    "local_crossover",
    "poisson",
    "smote",
    "replacement",
]:
    for split in range(1, 26):
        compute_synthetic_quality(
            data_path=Path(f"data/{method}/{CLASS_SIZE}/{split}"),
            save_dir=Path(f"data/{method}/{CLASS_SIZE}/{split}/results_cosine"),
        )
