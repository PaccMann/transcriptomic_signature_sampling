## TODO: needs refactoring and remove paths before committing to GitHub
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_dataset(
    data_paths: List,
    features_subset: ArrayLike,
    standardise: bool,
):
    df_list = []
    for path in data_paths:
        df_list.append(pd.read_csv(path, sep="\t", index_col=0).T)
    df = pd.concat(df_list)
    # remove version number if present (for rna-seq)
    if "." in df.columns[0]:
        df.columns = [col.split(".")[0] for col in df.columns]
    # drop non-primary tumours
    if len(df.index[0]) > 12:
        if len(df.index[0].rsplit("-", 1)[1]) == 3:
            idx_to_drop = [idx for idx in df.index if "-01A" not in idx]
            df.drop(index=idx_to_drop, inplace=True)
        elif len(df.index[0].rsplit("-", 1)[1]) == 2:
            idx_to_drop = [idx for idx in df.index if "-01" not in idx]
            df.drop(index=idx_to_drop, inplace=True)
    # stdz index
    df.index = [idx.rsplit("-", 1)[0] for idx in df.index]

    print("original df shape: ", df.shape)

    if features_subset is not None:
        column_ids = np.argwhere(df.columns.isin(features_subset))
        df = df.iloc[:, column_ids[:, 0]]
        print("subset df shape: ", df.shape)

    if standardise == True:
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    return df


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


probemap = pd.read_csv(
    "/Users/nja/Desktop/Sinergia/data/gdc_pancan/gene_expression/gencode.v22.annotation.gene.probeMap",
    sep="\t",
)
# probemap.index = probemap.id
# probemap = probemap.drop(columns='id')
probemap["length"] = probemap.chromEnd - probemap.chromStart
probemap = probemap.replace({"WARS": "WARS1"})
probemap = probemap.set_index("gene")

save_dir = Path("/Users/nja/Desktop/Sinergia/unicode_multiomics/data")

rnaseq_counts_coad = "/Users/nja/Desktop/Sinergia/data/gdc_coad/gene_expression/TCGA-COAD.htseq_counts.tsv.gz"
rnaseq_counts_read = "/Users/nja/Desktop/Sinergia/data/gdc_read/gene_expression/TCGA-READ.htseq_counts.tsv.gz"
dnameth_coadread_450k = "/Users/nja/Desktop/Sinergia/data/tcga_coadread/dna_methylation/HumanMethylation450.gz"
mirna_coad = "/Users/nja/Desktop/Sinergia/data/gdc_coad/TCGA-COAD.mirna.tsv.gz"
mirna_read = "/Users/nja/Desktop/Sinergia/data/gdc_read/TCGA-READ.mirna.tsv.gz"
rppa_coadread = "/Users/nja/Desktop/Sinergia/data/tcga_coadread/RPPA_RBN.gz"

## Process RNA-Seq

colotype_genes = pd.read_csv(
    "/Users/nja/Desktop/Sinergia/data/colotype_gex/colotype_genes.csv", index_col=0
)

rnaseq_counts_coad_df = pd.read_csv(rnaseq_counts_coad, index_col=0, sep="\t").T
rnaseq_counts_read_df = pd.read_csv(rnaseq_counts_read, index_col=0, sep="\t").T
rnaseq_counts_coadread_df = pd.concat([rnaseq_counts_coad_df, rnaseq_counts_read_df])
rnaseq_counts_coadread_df.columns = [
    col.split(".")[0] for col in rnaseq_counts_coadread_df.columns
]
idx_to_drop = [idx for idx in rnaseq_counts_coadread_df.index if "-01A" not in idx]
rnaseq_counts_coadread_df.drop(index=idx_to_drop, inplace=True)
rnaseq_counts_coadread_colotype_df = rnaseq_counts_coadread_df.loc[
    :, colotype_genes["ensemblid"]
]
rnaseq_counts_coadread_colotype_df.columns = colotype_genes["SYMBOL"]
gene_lengths = probemap["length"][rnaseq_counts_coadread_colotype_df.columns]
rnaseq_fpkm_coadread_colotype_df = get_fpkm_from_count(
    rnaseq_counts_coadread_colotype_df, gene_lengths, standardise=False
)
rnaseq_fpkm_coadread_colotype_df.index = [
    idx.rsplit("-", 1)[0] for idx in rnaseq_fpkm_coadread_colotype_df.index
]
rnaseq_fpkm_coadread_colotype_df.to_csv(
    save_dir / "rnaseq_fpkm_coadread_colotype_df.csv"
)

## Process RPPA

proteins_literature = [
    "BETACATENIN",
    "P53",
    "COLLAGENVI",
    "FOXO3A",
    "INPP4B",
    "PEA15",
    "PRAS40PT246",
    "RAD51",
    "S6",
    "S6PS235S236",
    "S6PS240S244",
]
rppa_coadread_literature11_df = get_dataset(
    [rppa_coadread], features_subset=proteins_literature, standardise=False
)

positive_scaler = MinMaxScaler()
df_scaled = positive_scaler.fit_transform(rppa_coadread_literature11_df)

rppa_coadread_literature11_df_scaled = pd.DataFrame(
    df_scaled,
    index=rppa_coadread_literature11_df.index,
    columns=rppa_coadread_literature11_df.columns,
)

rppa_coadread_literature11_df_scaled.to_csv(
    save_dir / "rppa_coadread_literature11_df_scaled.csv"
)
rppa_coadread_literature11_df.to_csv(save_dir / "rppa_coadread_literature11_df.csv")

## Process DNA Methylation
dna_probes = [
    "cg17477990",
    "cg11125249",
    "cg02827572",
    "cg04739880",
    "cg00512872",
    "cg14754494",
    "cg19107055",
    "cg05357660",
    "cg23045908",
    "cg16708174",
    "cg00901138",
    "cg00901574",
    "cg16477879",
    "cg23219253",
    "cg05211192",
    "cg12492273",
    "cg16772998",
    "cg00145955",
    "cg00097384",
    "cg27603796",
    "cg23418465",
    "cg17842966",
    "cg19335412",
    "cg23928468",
    "cg05951860",
    "cg20698769",
    "cg06786372",
    "cg17301223",
    "cg15638338",
    "cg02583465",
    "cg18065361",
    "cg06551493",
    "cg12691488",
    "cg17292758",
    "cg16170495",
    "cg21585512",
    "cg24702253",
    "cg17187762",
    "cg05983326",
    "cg11885357",
]
meth_genes = [
    "DUSP9",
    "GHSR",
    "MMP9",
    "PAQR9",
    "PRKG2",
    "PTH2R",
    "SLITRK4",
    "TIAM1",
    "SEMA7A",
    "GATA4",
    "LHX2",
    "SOST",
    "CTLA4",
    "NMNAT2",
    "ZFP42",
    "NPAS2",
    "MYLK3",
    "NUDT13",
    "KIRREL3",
    "FKBP6",
    "SOST",
    "NFATC1",
    "TLE4",
]
meth_gene_map = pd.read_csv(
    "/Users/nja/Downloads/illuminaMethyl27_hg38_GDC.tsv", sep="\t"
)
dna_probes_meth_genes = meth_gene_map[meth_gene_map["gene"].isin(meth_genes)][
    "#id"
].tolist()

dnameth_literature = dna_probes_meth_genes + dna_probes

dnameth_tcga_coadread_450_literature82 = get_dataset(
    [dnameth_coadread_450k], features_subset=dnameth_literature, standardise=False
)

dnameth_tcga_coadread_450_literature82.to_csv(
    save_dir / "dnameth_tcga_coadread_450_literature82.csv"
)

## Process miRNA
mirna_features = [
    "miR-17",
    "miR-18a",
    "miR-19a",
    "miR-19b",
    "miR-20a",
    "miR-21",
    "miR-27a",
    "miR-29a",
    "miR-91",
    "miR-92a-1",
    "miR-92a-2",
    "miR-135b",
    "miR-223",
    "miR-200a",
    "miR-200b",
    "miR-200c",
    "miR-141",
    "miR-143",
    "miR-145",
    "miR-221",
    "miR-222",
    "miR-99a",
    "miR-100",
    "miR-144",
    "miR-486-1",
    "miR-486-2",
    "miR-15b",
    "miR-1247",
    "miR-584",
    "miR-483",
    "miR-10a",
    "miR-425",
]
mirna_features = list(map(lambda x: ("hsa-" + x).lower(), mirna_features))

mirna_coadread_literature30_df = get_dataset(
    [mirna_coad, mirna_read], features_subset=mirna_features, standardise=False
)
mirna_coadread_literature30_df.to_csv(save_dir / "mirna_coadread_literature30_df.csv")


## Create multi-omics dataset
cologex_literature_files = [
    save_dir / "rnaseq_fpkm_coadread_colotype_df.csv",
    save_dir / "dnameth_tcga_coadread_450_literature82.csv",
    save_dir / "rppa_coadread_literature11_df_scaled.csv",
    save_dir / "mirna_coadread_literature30_df.csv",
]

df_list = []
for file in cologex_literature_files:
    df = pd.read_csv(file, index_col=0)
    df_list.append(df)

merged_cologex_literature_df = df_list[0].join(df_list[1:], how="outer")
merged_cologex_literature_df_innerjoin = df_list[0].join(df_list[1:], how="inner")

merged_cologex_literature_df = merged_cologex_literature_df.dropna(axis=1, how="all")
merged_cologex_literature_df.to_csv(save_dir / "merged_cologex_literature_df_v2.csv")

merged_cologex_literature_df_innerjoin = merged_cologex_literature_df_innerjoin.dropna(
    axis=1, how="all"
)
merged_cologex_literature_df_innerjoin.to_csv(
    save_dir / "merged_cologex_literature_df_v2_innerjoin.csv"
)

# ## Add WSI embeddings from VIT
# files_to_merge = [
#     "merged_cologex_literature_df_v2_innerjoin.csv",
#     "merged_cologex_literature_df_v2.csv",
# ]
# merge_how = {files_to_merge[0]: "inner", files_to_merge[1]: "outer"}
# vit_mean_patch = "/Users/nja/Desktop/Sinergia/data/coadread_data_clustering/vit-wsi_crc_mean_patch.csv"
# wsi_df = pd.read_csv(vit_mean_patch, index_col=0)
# for file in files_to_merge:
#     filename = "vit-wsi768_unstdz_" + file.split(".")[0] + ".csv"
#     df = pd.read_csv(save_dir / file, index_col=0)
#     wsi_merged_df = df.join(wsi_df, how=merge_how[file])
#     wsi_merged_df.to_csv(save_dir / filename)

## Generate Label File
clin_labels_coadread = pd.read_csv(
    "/Users/nja/Desktop/Sinergia/data/tinder_expts/tcga_colotype_all_labels_clean.csv",
    index_col=0,
)
intersect_idx_innerjoin = list(
    set(clin_labels_coadread.index) & set(merged_cologex_literature_df_innerjoin.index)
)
intersect_idx_outerjoin = list(
    set(clin_labels_coadread.index) & set(merged_cologex_literature_df.index)
)


clin_labels_innerjoin = clin_labels_coadread.loc[intersect_idx_innerjoin, :]
clin_labels_outerjoin = clin_labels_coadread.loc[intersect_idx_outerjoin, :]

merged_cologex_literature_df_innerjoin = merged_cologex_literature_df_innerjoin.loc[
    intersect_idx_innerjoin, :
]
merged_cologex_literature_df_outerjoin = merged_cologex_literature_df.loc[
    intersect_idx_outerjoin, :
]

assert all(clin_labels_innerjoin.index == merged_cologex_literature_df_innerjoin.index)
assert all(clin_labels_outerjoin.index == merged_cologex_literature_df_outerjoin.index)

clin_labels_innerjoin.to_csv(
    save_dir / "merged_cologex_literature_labels_innerjoin.csv"
)
merged_cologex_literature_df_innerjoin.to_csv(
    save_dir / "merged_cologex_literature_df_innerjoin.csv"
)
clin_labels_outerjoin.to_csv(
    save_dir / "merged_cologex_literature_labels_outerjoin.csv"
)
merged_cologex_literature_df_outerjoin.to_csv(
    save_dir / "merged_cologex_literature_df_outerjoin.csv"
)
