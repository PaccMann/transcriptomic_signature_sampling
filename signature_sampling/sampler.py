import math
import random
from collections import Counter
from typing import Any, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributions as td
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from signature_sampling.utils import time_func


class BaseSampler:
    """Sampling class to generate datasets augmented by various sampling strategies for
    Transcriptomic data."""

    def __init__(self, sampling_method: str, class_size: Any) -> None:
        """_summary_

        Args:
            sampling_method (str): Sampling method to use. One of ["poisson","gamma_poisson",
                "crossover_global","crossover_local","smote", "replacement"].
            class_size (int): Total size of the class after augmentation. For eg: if 500,
             # synthetic samples + # real samples = 500.
        """

        self.sampling_method = sampling_method
        self.class_size = class_size

        self.get_samples = {
            "unmod_poisson": self.poisson_unmod,
            "poisson": self.poisson_local_mean,
            "replacement": self.replacement_sampler,
            "gamma_poisson": self.gamma_poisson_sampler,
            "unmod_gamma_poisson": self.gamma_poisson_unmod,
        }

    def init_target_signatures(self, target_signatures: dict):
        self.target_signatures = target_signatures

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

    def poisson_unmod(
        self, X: pd.DataFrame, length: int, target_class: str, **kwargs
    ) -> pd.DataFrame:
        """Poisson sampling strategy for RNA-Seq data representing a phenotype (here, cms).

        Args:
            X (pd.DataFrame): Dataframe containing RNA-Seq values for patients of a
            given phenotype.
            Indices are patient ID and columns represent genes.
            length (int): Number of samples to generate.
            target_class (str): target class being augmented. Used only for creating a unique
            patient ID.

        Returns:
            pd.DataFrame: Dataframe of only newly generated samples from the Poisson
            distribution with the specified "cms" label.
        """
        samples = []
        project = X.index[0].split("-")[0]
        mean = X.mean().values
        poisson = list(map(td.Poisson, mean))
        for i in range(length):
            samples.append(list(map(lambda x: x.sample().item(), poisson)))

        return pd.DataFrame(
            samples,
            index=[f"{project}-{target_class}-S{i}" for i in range(length)],
            columns=X.columns,
        )

    def poisson_local_mean(
        self, X: pd.DataFrame, length: int, target_class: str, **kwargs
    ) -> pd.DataFrame:
        """Poisson sampling strategy for RNA-Seq data representing a phenotype (here, cms).

        Args:
            X (pd.DataFrame): Dataframe containing RNA-Seq values for patients of a
            given phenotype.
            Indices are patient ID and columns represent genes.
            length (int): Number of samples to generate.
            target_class (str): target class being augmented. Used only for creating a unique
            patient ID.

        Returns:
            pd.DataFrame: Dataframe of only newly generated samples from the Poisson
            distribution with the specified "cms" label.
        """
        samples = []
        project = X.index[0].split("-")[0]
        r_set_size = kwargs.get("poisson_r", 5)  # was 10
        ref_indices = self.random_combinations(
            X.index, subset_size=r_set_size, length=length
        )
        for i in range(length):
            mean = X.iloc[ref_indices[i], :].mean().values
            poisson = list(map(td.Poisson, mean))
            samples.append(list(map(lambda x: x.sample().item(), poisson)))

        return pd.DataFrame(
            samples,
            index=[f"{project}-{target_class}-S{i}" for i in range(length)],
            columns=X.columns,
        )

    def gamma_poisson_unmod(
        self, X: pd.DataFrame, length: int, target_class: str, **kwargs
    ) -> pd.DataFrame:
        """Gamma-Poisson (Negative Binomial) sampling strategy for RNA-Seq data
        representing a phenotype (here, cms).

        Args:
            X (pd.DataFrame): Dataframe containing RNA-Seq values for patients of a
            given phenotype.
            Indices are patient ID and columns represent genes.
            length (int): Number of samples to generate.
            target_class (str): target class being augmented. Used only for creating a unique
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
        project = X.index[0].split("-")[0]

        mean = X.mean().values
        var = X.var().values
        mu = mu_gammapoisson(mean, var)
        poisson = list(map(td.Poisson, mu))
        for i in range(length):
            samples.append(list(map(lambda x: x.sample().item(), poisson)))

        return pd.DataFrame(
            samples,
            index=[f"{project}-{target_class}-S{i}" for i in range(length)],
            columns=X.columns,
        )

    def gamma_poisson_sampler(
        self, X: pd.DataFrame, length: int, target_class: str, **kwargs
    ) -> pd.DataFrame:
        """Gamma-Poisson (Negative Binomial) sampling strategy for RNA-Seq data
        representing a phenotype (here, cms).

        Args:
            X (pd.DataFrame): Dataframe containing RNA-Seq values for patients of a
            given phenotype.
            Indices are patient ID and columns represent genes.
            length (int): Number of samples to generate.
            target_class (str): target class being augmented. Used only for creating a unique
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
        project = X.index[0].split("-")[0]
        r_set_size = kwargs.get("gamma_poisson_r", 5)
        ref_indices = self.random_combinations(
            X.index, subset_size=r_set_size, length=length
        )

        for i in range(length):
            mean = X.iloc[ref_indices[i], :].mean().values
            var = X.iloc[ref_indices[i], :].var().values
            mu = mu_gammapoisson(mean, var)
            poisson = list(map(td.Poisson, mu))
            samples.append(list(map(lambda x: x.sample().item(), poisson)))

        return pd.DataFrame(
            samples,
            index=[f"{project}-{target_class}-S{i}" for i in range(length)],
            columns=X.columns,
        )

    def replacement_sampler(
        self, X: pd.DataFrame, length: int, target_class: str, **kwargs
    ) -> pd.DataFrame:
        """Random oversampling method.

        Args:
            X (pd.DataFrame): Dataframe containing RNA-Seq values for patients of a
            given phenotype.
            Indices are patient ID and columns represent genes.
            length (int): Number of samples to generate.
            target_class (str): target class being augmented. Used only to match attribute signature of other functions.

        Returns:
            pd.DataFrame: Dataframe of only newly generated samples resampled from the
            given dataframe X.
        """
        random_idx = np.random.randint(0, len(X), size=length)
        ref_sample = X.iloc[random_idx, :]
        ref_sample.index = ref_sample.index + [f"S{i}" for i in range(length)]
        return ref_sample

    @time_func(method_name=lambda self: self.sampling_method)
    def sample(self, X: pd.DataFrame, y: pd.DataFrame, target: str, **kwargs) -> Tuple:
        """Main method to generate augmented datasets.

        Args:
            X (pd.DataFrame): Dataframe of transcriptomic data to be augmented.
            y (pd.DataFrame): Dataframe of target labels associated with the transcriptomic
            data.
            target (str): The name of the target labels, i.e, column name used in the
                dataframe for the targets. For example, 'cms', 'cimp', 'tStage', etc.
            dset_size (int): Desired size of each phenotype class. Same size applies
            to all classes.
            sampling_method (str, optional): Type of sampling to perform.
            One of ["poisson","replacement","gamma_poisson","smote","crossover_local","crossover_global"].
            Defaults to "gamma_poisson".

        Returns:
            Tuple: Tuple containing a dataframe of only the newly generated samples and
            a second dataframe of the associated target labels.
        """
        if self.sampling_method not in self.get_samples.keys():
            raise NotImplementedError("Sampling method unavailable.")

        target_count = dict(Counter(y[target]))
        sorted_count = sorted(target_count.items(), key=lambda item: item[1])
        # max_label_count = max(target_count.items(),key=lambda k: k[1])
        max_count = sorted_count[-1][1]

        sampled_df = pd.DataFrame()
        sampled_labels = pd.DataFrame()

        for k, v in sorted_count:
            if k in ["NOLBL", math.nan]:
                continue
            subset_idx = np.argwhere(y[target].values == k).flatten()
            subset = X.iloc[subset_idx, :]
            # size = max_count - v if self.class_size is None else self.class_size - v
            size = (
                max_count - v if self.class_size is None else self.class_size
            )  # if you want to sample specified dset_size samples

            new_samples = self.get_samples[self.sampling_method](
                subset, size, k, **kwargs
            )

            new_labels = pd.DataFrame({target: [k] * size}, index=new_samples.index)

            sampled_df = pd.concat([sampled_df, new_samples])
            sampled_labels = pd.concat([sampled_labels, new_labels])

        return sampled_df, sampled_labels


class SMOTESampler(BaseSampler):
    def __init__(self, sampling_method: str, class_size: int) -> None:
        super().__init__(sampling_method, class_size)

    @time_func(method_name="smote")
    def sample(self, X: pd.DataFrame, y: pd.DataFrame, target: str, **kwargs) -> Tuple:
        """Synthetic Minority Oversampling Technique.

        Args:
            X (pd.DataFrame): The entire dataset of RNA-Seq values (not restricted to a
            given phenotype) from which sampling should be done.
            y (pd.DataFrame): Target labels associated with the patients in X.
            target (str): The name of the target labels, i.e, column name used in the
                dataframe for the targets. For example, 'cms', 'cimp', 'tStage', etc.
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
        min_neighbours = min(Counter(y[target]).values())
        kneighbours = min(5, min_neighbours - 1)

        if self.class_size != None:
            sampling_strategy = dict.fromkeys(pd.unique(y[target]))
            for k in sampling_strategy.keys():
                sampling_strategy[k] = self.class_size
        else:
            sampling_strategy = "auto"

        sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=kneighbours)

        le_index = LabelEncoder()

        df_tmp = X.reset_index()
        df_tmp["index"] = le_index.fit_transform(df_tmp["index"])

        resampled_df, resampled_labels = sm.fit_resample(df_tmp, y)

        resampled_df[target] = resampled_labels
        resampled_df["index"] = le_index.inverse_transform(resampled_df["index"])

        resampled_df = resampled_df.set_index("index")
        data_labels = X.join(y)

        synthetic_samples = pd.concat([resampled_df, data_labels]).drop_duplicates(
            keep=False
        )
        synthetic_samples["index"] = synthetic_samples.index
        synthetic_samples.index = (
            synthetic_samples.index
            + "S"
            + synthetic_samples.groupby("index").cumcount().astype(str)
        )

        synthetic_labels = pd.DataFrame(synthetic_samples[target])

        synthetic_samples = synthetic_samples.drop(columns=[target, "index"])

        # merged_df = pd.concat([X, synthetic_samples])
        # synthetic_labels
        # merged_labels = pd.concat([y, synthetic_labels])

        # return merged_df, merged_labels

        return synthetic_samples, synthetic_labels
