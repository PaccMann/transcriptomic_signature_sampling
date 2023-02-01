import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

seeds = [42, 182, 327]
num_repeats = 3

parser = argparse.ArgumentParser()

parser.add_argument(
    "root_dir", type=str, help="Path to the main directory containing the datasets."
)

parser.add_argument(
    "results_dir", type=str, help="Path to save the results, logs and best model."
)

parser.add_argument(
    "sampling_method",
    type=str,
    help="Sampling method to be tested with the model. Enter data folder name.",
)

parser.add_argument(
    "size",
    type=str,
    help="class-size of augmented data. Used to access the right folder to load the data. Enter "
    " if None.",
)


def main(root_dir, results_dir, sampling_method, size):

    summary_clf = []
    for repeat in range(num_repeats):
        results = dict()
        torch.manual_seed(seeds[repeat])
        np.random.seed(seeds[repeat])

        data_dir = os.path.join(root_dir, sampling_method, size)

        save_dir = os.path.join(results_dir, sampling_method, size, f"run_{repeat}")
        os.makedirs(save_dir, exist_ok=True)

        x_train = pd.read_csv(
            os.path.join(data_dir, "train_logfpkm_colotype_stdz.csv"), index_col=0
        )
        y_train = pd.read_csv(
            os.path.join(data_dir, "train_labels_colotype.csv"), index_col=0
        )
        x_test = pd.read_csv(
            os.path.join(data_dir, "test_logfpkm_colotype_stdz.csv"), index_col=0
        )
        y_test = pd.read_csv(
            os.path.join(data_dir, "test_labels_colotype_stdz.csv"), index_col=0
        )

        clf = RandomForestClassifier()
        acc_cv = cross_val_score(
            clf, x_train, y_train, cv=5, scoring="balanced_accuracy"
        )

        results["balanced_acc_cv"] = list(acc_cv)
        results["cv_mean"] = np.mean(acc_cv)
        results["cv_std"] = np.std(acc_cv)

        clf.fit(x_train, y_train)
        y_preds = clf.predict(x_test)
        joblib.dump(clf, os.path.join(save_dir, "rf_model"))
        node_indicator, _ = clf.decision_path(x_test)
        np.save(os.path.join(save_dir, "test_decision_path"), node_indicator)

        test_acc = sm.balanced_accuracy_score(y_test, y_preds)
        results["test_acc"] = test_acc

        clf_report = classification_report_imbalanced(
            y_test,
            y_preds,
            target_names=["CMS1", "CMS2", "CMS3", "CMS4"],
            output_dict=True,
            zero_division=0,
        )
        clf_report_df = pd.DataFrame.from_dict(clf_report).T
        clf_report_df.to_csv(os.path.join(save_dir, "clf_report.csv"))
        summary_clf.append(clf_report_df)

        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f)

    df_concat = pd.concat(summary_clf)
    df_grouped = df_concat.groupby(df_concat.index)
    df_means = df_grouped.mean()
    df_std = df_grouped.std()

    df_means.to_csv(
        os.path.join(results_dir, sampling_method, size, "clf_report_mean.csv")
    )
    df_std.to_csv(
        os.path.join(results_dir, sampling_method, size, "clf_report_std.csv")
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.root_dir, args.results_dir, args.sampling_method, args.size)
