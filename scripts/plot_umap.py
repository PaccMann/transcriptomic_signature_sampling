import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import click
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from umap import UMAP

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plot_umap(
    sampling_method,
    ax_,
    data,
    labels,
    hue,
    hue_order,
    umap_min_dist,
    umap_neighbours,
    **kwargs,
):
    if sampling_method == "smote":
        reducer = UMAP(
            min_dist=umap_min_dist,
            random_state=42,
            n_neighbors=umap_neighbours,
            unique=True,
            **kwargs,
        )
    else:
        reducer = UMAP(
            min_dist=umap_min_dist,
            random_state=42,
            n_neighbors=umap_neighbours,
            **kwargs,
        )

    sns.set(font_scale=1.0)
    sns.set_style("whitegrid", {"axes.grid": False})
    embedding = reducer.fit_transform(data)

    plot_df = pd.DataFrame(index=data.index)

    plot_df["UMAP1"] = embedding[:, 0]
    plot_df["UMAP2"] = embedding[:, 1]

    plot_df = plot_df.join(labels)

    plot_df["data"] = [
        "synthetic" if re.search("S\d", index) is not None else "real"
        for index in plot_df.index
    ]

    s = [100 if item == "synthetic" else 75 for item in plot_df["data"]]
    alphas = [1.0 if item == "synthetic" else 0.8 for item in plot_df["data"]]
    # alphas=0.8
    plot_df["size"] = s
    num_label_type = len(plot_df[hue].unique())
    current_palette = (
        sns.color_palette("colorblind")
        + sns.color_palette("dark")
        + sns.color_palette("deep")
    )
    current_palette = current_palette[0:num_label_type]

    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        data=plot_df[plot_df["data"] == "synthetic"],
        hue=hue,
        palette=current_palette,
        style="data",
        hue_order=hue_order,
        sizes=s,
        markers={"synthetic": ".", "real": "X"},
        alpha=0.5,
        ax=ax_,
    )

    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        data=plot_df[plot_df["data"] == "real"],
        hue=hue,
        palette=current_palette,
        style="data",
        hue_order=hue_order,
        sizes=s,
        markers={"synthetic": ".", "real": "X"},
        alpha=1.0,
        ax=ax_,
    )
    ax_.get_legend().remove()

    # uncomment only for size Max.
    # if sampling_method == "smote":
    #     axin = ax_.inset_axes([0.1, 0.7, 0.25, 0.25])
    #     # axin = ax_.inset_axes([0.7, 0.05, 0.25, 0.25])
    #     sns.scatterplot(
    #         x="UMAP1",
    #         y="UMAP2",
    #         data=plot_df[plot_df["data"]=="synthetic"],
    #         hue=hue,
    #         palette=current_palette,
    #     style="data",
    #     hue_order=hue_order,
    #     sizes=s,
    #     markers={"synthetic": ".", "real": "X"},
    #     alpha=0.7,
    #     ax=axin
    # )

    # sns.scatterplot(
    #     x="UMAP1",
    #     y="UMAP2",
    #     data=plot_df[plot_df["data"]=="real"],
    #     hue=hue,
    #     palette=current_palette,
    #     style="data",
    #     hue_order=hue_order,
    #     sizes=s,
    #     markers={"synthetic": ".", "real": "X"},
    #     alpha=1.0,
    #     ax=axin
    # )
    # 5,10 - max
    # 2.5, 7 - 500
    # 6.5,11.5 - 5000
    # axin.set_xlim(3,4)
    # #10.5,12.5 - max
    # #7.5,10 - 500
    # #0,2.5 - 5000
    # # axin.set_ylim(9.0,11.0)
    # axin.set_ylim(4,5)
    # axin.get_legend().remove()
    # ax_.indicate_inset_zoom(axin)
    # axin.set(xlabel=None,ylabel=None)
    # axin.set_xticks([])
    # axin.set_yticks([])


@click.command()
@click.option(
    "--sampling_method",
    default="gamma_poisson",
)
@click.option("--size", type=str)
@click.option("--data_dir", type=click.Path(path_type=Path, exists=True))
@click.option("--save_dir", type=click.Path(path_type=Path))
def main(sampling_method, size, data_dir, save_dir):
    current_palette = (
        sns.color_palette("colorblind")
        + sns.color_palette("dark")
        + sns.color_palette("deep")
    )
    labels = ["CMS1", "CMS2", "CMS3", "CMS4", "Synthetic Data", "Real Data"]
    hue_order = ["CMS1", "CMS2", "CMS3", "CMS4"]
    # labels = ["LumA", "LumB", "Basal", "Synthetic Data", "Real Data"]
    # hue_order = ["LumA", "LumB", "Basal"]

    patches = []
    for i, colour in enumerate(current_palette[0 : len(hue_order)]):
        patches.append(
            mpatches.Patch(
                color=colour,
                label=labels[i],
            )
        )
    patches.append(
        Line2D(
            [0],
            [0],
            label="Synthetic Data",
            marker=".",
            markersize=10,
            markerfacecolor="k",
            linestyle="",
        )
    )
    patches.append(
        Line2D(
            [0],
            [0],
            label="Real Data",
            marker="X",
            markersize=10,
            markeredgecolor=None,
            markerfacecolor="k",
            linestyle="",
        )
    )
    # labels = ["CMS1", "CMS2", "CMS3", "CMS4", "Synthetic Data", "Real Data"]

    subplot_title_map = dict(zip(sampling_method, "Mod. Gamma-Poisson"))

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(18, 5), sharex=True, sharey=True)
    # for i, sampling in enumerate(sampling_methods):
    #     if sampling == "unaugmented":
    #         class_size = ""
    #     else:
    #         class_size = size
    #     train_df = pd.read_csv(
    #         data_dir
    #         / sampling
    #         / class_size
    #         / split
    #         / "metabric_stdz.csv",
    #         index_col=0,
    #     )
    #     train_labels = pd.read_csv(
    #         data_dir
    #         / sampling
    #         / class_size
    #         / split
    #         / "metabric_labels.csv",
    #         index_col=0,
    #     )

    #     concat_df = train_df
    #     concat_labels = train_labels

    if sampling_method == "unaugmented":
        class_size = ""
    else:
        class_size = size
    train_df = pd.read_csv(
        data_dir / class_size / "augmented_train_df_stdz.csv",
        index_col=0,
    )
    train_labels = pd.read_csv(
        data_dir / class_size / "augmented_train_labels.csv",
        index_col=0,
    )
    test_df = pd.read_csv(
        data_dir / class_size / "real_test_df_stdz.csv",
        index_col=0,
    )
    test_labels = pd.read_csv(
        data_dir / class_size / "real_test_labels.csv",
        index_col=0,
    )
    concat_df = pd.concat([train_df, test_df])
    concat_labels = pd.concat([train_labels, test_labels])

    plot_umap(
        sampling_method,
        axs,
        concat_df,
        concat_labels,
        "cms",  # cms PAM50 Pam50 + Claudin-low subtype
        hue_order,
        0.1,  # 0.1 or 0.5
        5,
        # **{"init": "pca", "n_components": 10},
    )

    fig.suptitle(
        "Visualisation of mod. GP augmented and real data",
        fontsize=16,
    )
    fig.legend(
        patches,
        labels,
        loc="lower center",
        ncols=7,
        prop={"size": 14},
        shadow=False,
        bbox_to_anchor=(0.5, -0.1),
    )
    save_dir = data_dir / class_size
    fig.savefig(save_dir / "umap.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
