import random
from collections import Counter
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch.distributions as td
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


class Sampler:
    """Sampling class to generate datasets augmented by various sampling strategies for
    Transcriptomic data."""

    def __init__(self) -> None:
        """Constructor class"""
        self.get_samples = {
            "poisson": self.poisson_local_mean,
            "replacement": self.replacement_sampler,
            "crossover_global": self.crossover_sampler_global,
            "crossover_local": self.crossover_sampler_local,
            "smote": self.smote_sampler,
            "gamma_poisson": self.gamma_poisson_sampler,
        }

    def random_combinations(
        self, indices: Iterable, subset_size: int, length: int
    ) -> List:
        """Random selection of subsets of indices.

        Args:
            indices (Iterable): The set of indices from which subsets should be created.
            subset_size (int): Size of the subset.
            length (int): Number of subsets to create.

        Returns:
            List: List of subsets as a tuple.
        """
        "Random selection from itertools.combinations(iterable, r)"
        n = len(indices)
        grouped_indices = [
            sorted(random.sample(range(n), subset_size)) for i in range(length)
        ]
        return grouped_indices

    def poisson_local_mean(
        self, X: pd.DataFrame, length: int, cms: str
    ) -> pd.DataFrame:
        """Poisson sampling strategy for RNA-Seq data representing a phenotype (here, cms).

        Args:
            X (pd.DataFrame): Dataframe containing RNA-Seq values for patients of a
            given phenotype.
            Indices are patient ID and columns represent genes.
            length (int): Number of samples to generate.
            cms (str): CMS type being augmented. Used only for creating a unique
            patient ID.

        Returns:
            pd.DataFrame: Dataframe of only newly generated samples from the Poisson
            distribution with the specified "cms" label.
        """
        samples = []
        ref_indices = self.random_combinations(X.index, r=10, length=length)
        for i in range(length):
            mean = X.iloc[ref_indices[i], :].mean().values
            poisson = list(map(td.Poisson, mean))
            samples.append(list(map(lambda x: x.sample().item(), poisson)))

        return pd.DataFrame(
            samples,
            index=[f"TCGA-{cms}-{i}S" for i in range(length)],
            columns=X.columns,
        )

    def gamma_poisson_sampler(
        self, X: pd.DataFrame, length: int, cms: str
    ) -> pd.DataFrame:
        """Gamma-Poisson (Negative Binomial) sampling strategy for RNA-Seq data
        representing a phenotype (here, cms).

        Args:
            X (pd.DataFrame): Dataframe containing RNA-Seq values for patients of a
            given phenotype.
            Indices are patient ID and columns represent genes.
            length (int): Number of samples to generate.
            cms (str): CMS type being augmented. Used only for creating a unique
            patient ID.

        Returns:
            pd.DataFrame: Dataframe of only newly generated samples from the
            Gamma-Poisson distribution
            with the specified "cms" label.
        """

        def mu_gammapoisson(mean: np.ndarray, var: np.ndarray) -> List:
            """Function sampling the mu parameter from the Gamma-Poisson distribution.

            Args:
                mean (np.ndarray): Array of means to initialise the Gamma distribution.
                var (np.ndarray): Array of standard deviations to initialise the Gamma
                distribution.

            Returns:
                List: List of means to intialise the Poisson distribution.
            """

            beta = mean / var
            alpha = mean * beta
            gamma = list(map(td.Gamma, alpha, beta))
            mu = list(map(lambda x: x.sample(), gamma))
            return mu

        samples = []
        ref_indices = self.random_combinations(X.index, r=5, length=length)

        for i in range(length):
            mean = X.iloc[ref_indices[i], :].mean().values
            var = X.iloc[ref_indices[i], :].var().values
            mu = mu_gammapoisson(mean, var)
            poisson = list(map(td.Poisson, mu))
            samples.append(list(map(lambda x: x.sample().item(), poisson)))

        return pd.DataFrame(
            samples,
            index=[f"TCGA-{cms}-{i}S" for i in range(length)],
            columns=X.columns,
        )

    def replacement_sampler(self, X: pd.DataFrame, length: int) -> pd.DataFrame:
        """Random oversampling method.

        Args:
            X (pd.DataFrame): Dataframe containing RNA-Seq values for patients of a
            given phenotype.
            Indices are patient ID and columns represent genes.
            length (int): Number of samples to generate.

        Returns:
            pd.DataFrame: Dataframe of only newly generated samples resampled from the
            given dataframe X.
        """
        random_idx = np.random.randint(0, len(X), size=length)
        ref_sample = X.iloc[random_idx, :]
        ref_sample.index = ref_sample.index + "S"
        return ref_sample

    def smote_sampler(
        self, X: pd.DataFrame, y: pd.DataFrame, sampling_strategy: dict
    ) -> Tuple:
        """Synthetic Minority Oversampling Technique.

        Args:
            X (pd.DataFrame): The entire dataset of RNA-Seq values (not restricted to a
            given phenotype) from which sampling should be done.
            y (pd.DataFrame): Target labels associated with the patients in X.
            sampling_strategy (dict): Argument passed to the smote method from the
            imblearn package.
            In this function, it is a dictionary with keys = cms types,
            values = desired size of that cms class.
            For more details on this argument, refer to https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html.

        Returns:
            Tuple: Tuple consisting of the augmented dataframe containing both real and
            newly generated samples, and a second dataframe containing the labels
            associated with these samples.
        """

        sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5)

        le_index = LabelEncoder()

        df_tmp = X.reset_index()
        df_tmp["index"] = le_index.fit_transform(df_tmp["index"])

        resampled_df, resampled_labels = sm.fit_resample(df_tmp, y)

        resampled_df["cms"] = resampled_labels
        resampled_df["index"] = le_index.inverse_transform(resampled_df["index"])

        resampled_df = resampled_df.set_index("index")
        data_labels = X.join(y)

        synthetic_samples = pd.concat([resampled_df, data_labels]).drop_duplicates(
            keep=False
        )
        synthetic_samples.index = synthetic_samples.index + "S"
        synthetic_labels = pd.DataFrame(synthetic_samples["cms"])

        synthetic_samples = synthetic_samples.drop(columns="cms")

        # merged_df = pd.concat([X, synthetic_samples])
        # synthetic_labels
        # merged_labels = pd.concat([y, synthetic_labels])

        # return merged_df, merged_labels

        return synthetic_samples, synthetic_labels

    def crossover_sampler_global(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        cms_size: Union[int, list, dict],
        colotype_genes: pd.DataFrame,
    ) -> Tuple:

        """Inter-Class Crossover sampling method. Sampling inspired by chromosomal
        crossing over where there is a "crossover" between samples at the gene signature
        level (associated with different CMS types). Global version where signatures are
        sampled from all samples except those samples from the class it predicts.

        Args:
            X (pd.DataFrame): The entire dataset of RNA-Seq values (not restricted to a
            given phenotype) from which sampling should be done.
            y (pd.DataFrame): Target labels associated with the patients in X.
            cms_size (Union[int, list, dict]): Desired size of each CMS type. The same
            size applies to all CMS types if an integer is provided.
            If a list is provided, ensure that the values correspond to the sorted CMS
            types,i.e, CMS1,CMS2,CMS3,CMS4.
            colotype_genes (pd.DataFrame): Dataframe of predictive/signature genes and
            the phenotype each gene is associated with.

        Returns:
            Tuple: Tuple containing a dataframe of only the newly generated samples and
            a second dataframe of the associated target labels.
        """

        target_count = dict(Counter(y["cms"]))
        sorted_count = sorted(target_count.items(), key=lambda item: item[1])

        sorted_count = sorted(target_count.items(), key=lambda item: item[1])
        max_count = sorted_count[-1][1]

        if any("ENS" in x for x in X.columns):
            gene_type = "ensemblid"
        else:
            gene_type = "SYMBOL"

        cms_genes = dict(
            colotype_genes.groupby("subtype").apply(lambda x: x[gene_type].values)
        )

        df = X.join(y)
        ref_df = df.sort_values(by="cms")
        cms_ids_all = ref_df.index
        cms_ids = {k: ref_df[ref_df["cms"] == k].index for k in np.unique(y)}
        cms_list = list(cms_ids.keys())
        sampling_idx = pd.DataFrame(columns=cms_list)
        sampled_labels = pd.DataFrame(columns=["cms"])
        row_idx = []
        sampled_df_list = []

        if isinstance(cms_size, int):
            cms_size_dict = dict.fromkeys(cms_list, cms_size)
        elif isinstance(cms_size, list):
            cms_key_values = list(zip(cms_list, cms_size))
            cms_size_dict = dict(cms_key_values)
        elif isinstance(cms_size, dict):
            cms_size_dict = cms_size

        # NOTE: doing this row wise per cms-block, so we sample all indices for cms1
        size_list = []  # - to keep track of cms sizes for indexing in new augmented df
        for cms in cms_list:

            row_idx = []

            size = (
                max_count - target_count[cms]
                if cms_size is None
                else cms_size_dict[cms]
                - target_count[
                    cms
                ]  # remove "- target_count[cms]" if you want to sample specified cms_size samples
            )
            size_list.append(size)
            if size == 0:
                continue

            for cms_ in cms_list:
                if cms == cms_:
                    target_cms = np.random.choice(cms_ids[cms], size=size)
                    row_idx.append(target_cms.tolist())

                else:
                    fragment = np.random.choice(
                        np.setdiff1d(cms_ids_all, cms_ids[cms_]), size
                    )
                    row_idx.append(fragment.tolist())

            sampled_cms = pd.DataFrame(row_idx, index=cms_list).T
            sampling_idx = pd.concat([sampling_idx, sampled_cms])
            new_labels = pd.DataFrame({"cms": [cms] * size})
            sampled_labels = pd.concat([sampled_labels, new_labels])

        sampling_idx = sampling_idx.reset_index(drop=True)
        sampled_labels = sampled_labels.reset_index(drop=True)

        ref_sample_index = []

        for i, cms in enumerate(cms_list):
            j = 0
            assert sum(sampling_idx[cms].isin(cms_ids[cms])) == size_list[i]
            if size_list[i] == 0:
                continue
            sample_indices = sampling_idx.loc[
                size_list[i] * j : (size_list[i] * (j + 1)) - 1, cms
            ]
            if sample_indices.dtype == int:
                sample_indices = sample_indices.astype(str)

            sample_indices = sample_indices + "S"

            ref_sample_index.append(sample_indices)
            j += 1

        for k, v in cms_genes.items():
            sampled_df_list.append(
                ref_df.loc[sampling_idx.loc[:, k].values, v].reset_index()
            )

        sampled_df = pd.concat(sampled_df_list, axis=1).drop(columns="index")

        sampled_df.index = [idx for subidx in ref_sample_index for idx in subidx]
        sampled_df.index.name = "id"
        sampled_df.index = sampled_df.index.astype(str) + sampled_df.groupby(
            "id"
        ).cumcount().astype(str)
        sampled_df = sampled_df.loc[:, X.columns]

        sampled_labels.index = sampled_df.index

        return sampled_df, sampled_labels

    def crossover_sampler_local(
        self, X: pd.DataFrame, length: int, cms: str, colotype_genes: pd.DataFrame
    ) -> pd.DataFrame:
        """Intra-Class Crossover sampling method. Sampling inspired by chromosmal
        crossing over where there is a "crossover" between samples at the gene signature
        level (associated with different CMS types). Local version where a subset of a
        given class is passed and the intermixing takes place only between samples of
        this class.
        Args:
            X (pd.DataFrame): Dataframe containing RNA-Seq values for patients of a
            given phenotype.
            length (int): Number of samples to generate.
            cms (str): CMS type being augmented. Used only for creating a unique
            patient ID.
            colotype_genes (pd.DataFrame): Dataframe of predictive/signature genes and
            the phenotype each gene is associated with.

        Returns:
            pd.DataFrame: Dataframe of only newly generated samples from the given
            dataframe X.
        """
        # check if X has gene name or ensembl id
        if any("ENS" in x for x in X.columns):
            gene_type = "ensemblid"
        else:
            gene_type = "SYMBOL"

        cms_genes = dict(
            colotype_genes.groupby("subtype").apply(lambda x: x[gene_type].values)
        )
        # here subset is set of all samples of a given class type
        cms_list = ["CMS1", "CMS2", "CMS3", "CMS4"]

        sampling_idx = np.random.randint(0, len(X), size=(length, 4))
        sampling_idx = pd.DataFrame(sampling_idx, columns=cms_list)
        sampled_df_list = []

        ref_sample_index = X.index[sampling_idx[cms]]
        if ref_sample_index.dtype == int:
            ref_sample_index = ref_sample_index.astype(str)
        ref_sample_index = ref_sample_index + "S"

        for k, v in cms_genes.items():
            idx = X.index[sampling_idx.loc[:, k]]
            sampled_df_list.append(X.loc[idx, v].reset_index())

        sampled_df = pd.concat(sampled_df_list, axis=1).drop(columns="index")
        sampled_df.index = ref_sample_index
        sampled_df.index.name = "id"
        sampled_df.index = sampled_df.index.astype(str) + sampled_df.groupby(
            "id"
        ).cumcount().astype(str)
        return sampled_df

    def subset_sampler(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        dset_size: int,
        sampling_method: str = "gamma_poisson",
        **kwargs,
    ) -> Tuple:
        """Main method to generate augmented datasets.

        Args:
            X (pd.DataFrame): Dataframe of transcriptomic data to be augmented.
            y (pd.DataFrame): Array of target labels associated with the transcriptomic
            data.
            dset_size (int): Desired size of each phenotype class. Same size applies
            to all classes.
            type (str, optional): Type of sampling to perform.
            One of ["poisson","replacement","gamma_poisson","smote","crossover_local","crossover_global"].
            Defaults to "gamma_poisson".

        Returns:
            Tuple: Tuple containing a dataframe of only the newly generated samples and
            a second dataframe of the associated target labels.
        """
        target_count = dict(Counter(y["cms"]))
        sorted_count = sorted(target_count.items(), key=lambda item: item[1])
        # max_label_count = max(target_count.items(),key=lambda k: k[1])
        max_count = sorted_count[-1][1]

        sampled_df = pd.DataFrame()
        sampled_labels = pd.DataFrame()

        for k, v in sorted_count:
            if k == "NOLBL":
                continue
            subset_idx = np.argwhere(y["cms"].values == k).flatten()
            subset = X.iloc[subset_idx, :]
            size = max_count - v if dset_size is None else dset_size - v
            # size = max_count - v if dset_size is None else dset_size # if you want to sample specified dset_size samples

            if sampling_method == "crossover_local":
                new_samples = self.get_samples[sampling_method](
                    subset, size, k, kwargs["colotype_genes"]
                )
                new_samples = new_samples.loc[:, X.columns]

            elif sampling_method in ["gamma_poisson", "poisson_local"]:
                new_samples = self.get_samples[sampling_method](subset, size, k)
            else:
                new_samples = self.get_samples[sampling_method](subset, size, **kwargs)

            new_labels = pd.DataFrame({"cms": [k] * size}, index=new_samples.index)

            sampled_df = sampled_df.append(new_samples)
            sampled_labels = sampled_labels.append(new_labels)

        return sampled_df, sampled_labels
