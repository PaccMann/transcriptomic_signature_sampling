import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


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
