from ast import literal_eval
from collections import Counter
from pathlib import Path
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import false_discovery_control, wilcoxon

from signature_sampling.crossover_sampling import CrossoverSampling
from signature_sampling.utils import significance_testing

sampling_method_keys = {
    "gamma_poisson": "Mod.GP",
    "global_crossover": "Inter-Class",
    "local_crossover": "Intra-Class",
    "poisson": "Mod.Poisson",
    "replacement": "Resampling",
    "smote": "SMOTE",
    "unaugmented": "Unaugmented",
}

colour_pal = sns.color_palette("colorblind")
hue_order_ = [
    "gamma_poisson",
    "global_crossover",
    "local_crossover",
    "poisson",
    "replacement",
    "smote",
    "unaugmented",
]
methods = [
    "Mod.GP",
    "Inter-Class",
    "Intra-Class",
    "Mod.Poisson",
    "Replacement",
    "SMOTE",
    "Unaugmented",
]
sns.set_palette(colour_pal)
sns.set(font="Helvetica")


@click.command()
@click.option(
    "--result_main_dir",
    type=Path,
    help="Main result directory containing all classifiers' results",
)
@click.option(
    "--classifiers",
    type=str,
    help="Pass a list of classifiers as a string. Eg '[]'",
    default='["Logistic", "SVM-RBF", "KNN", "RF", "EBM"]',
)
@click.option(
    "--sampling_methods",
    type=str,
    help="Pass a list of sampling methods as a string. Eg '[]'",
    default='["gamma_poisson","global_crossover","local_crossover","poisson","replacement","smote","unaugmented"]',
)
@click.option(
    "--class_sizes",
    type=str,
    help="Pass a list of class sizes as a string. Eg '[]'",
    default='["max","500","5000"]',
)
@click.option(
    "--internal_key",
    type=str,
    help="column name of in-domain test metric.",
    default="bal_acc",
)
@click.option(
    "--external_key",
    type=str,
    help="column name of external test metric.",
    default="cptac_bal_acc",
)
@click.option(
    "--save_dir", type=Path, help="Path where results and figures are to be saved."
)
def main(
    result_main_dir,
    classifiers,
    sampling_methods,
    class_sizes,
    internal_key,
    external_key,
    save_dir,
):
    # TODO - for auto num of sublots do it based on len of class sizes
    classifiers = literal_eval(classifiers)
    sampling_methods = literal_eval(sampling_methods)
    class_sizes = literal_eval(class_sizes)

    fig, axs = plt.subplots(1, len(class_sizes), sharey=True, figsize=(15, 4))
    ext_fig, ext_axs = plt.subplots(1, 3, sharey=True, figsize=(15, 4))

    for i, size in enumerate(class_sizes):
        overall_metrics_internal_list = []
        overall_metrics_external_list = []
        for method in sampling_methods:  #'MLP',
            for clf in classifiers:
                if method == "unaugmented":
                    result_path = result_main_dir / clf / method / "metrics.csv"
                    # result_path = f"/Users/nja/Desktop/Sinergia/results/tinder_expts/5x5stratified_v2_check/{clf}/{method}/metrics.csv"
                else:
                    result_path = result_main_dir / clf / method / size / "metrics.csv"
                    # result_path = f"/Users/nja/Desktop/Sinergia/results/tinder_expts/5x5stratified_v2_check/{clf}/{method}/{size}/metrics.csv"
                tmp_df = pd.read_csv(result_path)
                tmp_df["method"] = [method] * len(tmp_df)
                overall_metrics_internal_list.append(tmp_df[[internal_key, "method"]])
                overall_metrics_external_list.append(tmp_df[[external_key, "method"]])
        plot_bal_acc = pd.concat(overall_metrics_internal_list, axis=0)
        plot_ext_bal_acc = pd.concat(overall_metrics_external_list, axis=0)

        sns.violinplot(data=plot_bal_acc, x="method", y=internal_key, ax=axs[i])
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].set_xlabel("Data Augmentation Methods")
        axs[i].set_ylabel("Balanced Accuracy")
        axs[i].set_title(f"Class Size : {size}")
        # axs[i].set_ylim([0.7,0.9])
        axs[i].set_xticklabels(methods, rotation=45, ha="right", rotation_mode="anchor")
        fig.savefig(
            save_dir / f"violin_internal_{internal_key}.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        sns.violinplot(data=plot_ext_bal_acc, x="method", y=external_key, ax=ext_axs[i])
        handles, labels = ext_axs[i].get_legend_handles_labels()
        ext_axs[i].set_xlabel("Data Augmentation Methods")
        ext_axs[i].set_ylabel("External Balanced Accuracy")
        ext_axs[i].set_title(f"Class Size : {size}")
        # ext_axs[i].set_ylim([0.3,0.9])
        ext_axs[i].set_xticklabels(
            methods, rotation=45, ha="right", rotation_mode="anchor"
        )
        ext_fig.savefig(
            save_dir / f"violin_external_{internal_key}.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        w_df = pd.DataFrame(columns=sampling_methods, index=sampling_methods)
        w_df_ext = pd.DataFrame(columns=sampling_methods, index=sampling_methods)
        for smethod in sampling_methods:
            for cmethod in sampling_methods:
                if smethod == cmethod:
                    continue
                sdf = plot_bal_acc[plot_bal_acc["method"] == smethod]
                cdf = plot_bal_acc[plot_bal_acc["method"] == cmethod]

                w = wilcoxon(
                    sdf[internal_key], cdf[internal_key], alternative="greater"
                )

                w_df.loc[smethod, cmethod] = w.pvalue

                sdf_ext = plot_ext_bal_acc[plot_bal_acc["method"] == smethod]
                cdf_ext = plot_ext_bal_acc[plot_bal_acc["method"] == cmethod]

                w_ext = wilcoxon(
                    sdf_ext[external_key],
                    cdf_ext[external_key],
                    alternative="greater",
                )

                w_df_ext.loc[smethod, cmethod] = w_ext.pvalue

        w_df = pd.DataFrame(
            false_discovery_control(w_df.fillna(1)),
            index=w_df.index,
            columns=w_df.index,
        )
        w_df_ext = pd.DataFrame(
            false_discovery_control(w_df_ext.fillna(1)),
            index=w_df_ext.index,
            columns=w_df_ext.index,
        )
        w_df.to_csv(save_dir / f"internal_{internal_key}_p_values_{size}.csv")
        w_df_ext.to_csv(save_dir / f"external_{internal_key}_p_values_{size}.csv")


if __name__ == "__main__":
    main()
