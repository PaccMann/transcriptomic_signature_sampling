import math
import random
from collections import Counter
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch.distributions as td
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from signature_sampling.sampler import BaseSampler


class CrossoverSampling(BaseSampler):
    def __init__(self, sampling_method: str, class_size: int):
        """_summary_

        Args:
            sampling_method (str): Type of sampling to perform. One of ["crossover_local","crossover_global"]
            class_size (int): (Union[int, list, dict]): Desired size of each target class. The same
            size applies to all target classes if an integer is provided.
            NOTE: specifying class size 500 returns 500 new samples. If total class size including 500 is desired, then
                class_size = 500 - # real samples.
        """
        super().__init__(sampling_method, class_size)
        # self.target_signatures = target_signatures
        self.get_samples.update(
            {
                "local_crossover": self.intraclass_sampling,
                "global_crossover": self.interclass_sampling,
            }
        )

    def interclass_sampling(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        target: str,
    ) -> Tuple:
        """Inter-Class Crossover sampling method. Sampling inspired by chromosomal
        crossing over where there is a "crossover" between samples at the gene signature
        level (associated with different CMS types). Global version where signatures are
        sampled from all samples except those samples from the class it predicts.

        Args:
            X (pd.DataFrame): The entire dataset of RNA-Seq values (not restricted to a
            given phenotype) from which sampling should be done.
            y (pd.DataFrame): Target labels associated with the patients in X.
            target (str): The name of the target labels, i.e, column name used in the
                dataframe for the targets. For example, 'cms', 'cimp', 'tStage', etc.
            target_size (Union[int, list, dict]): Desired size of each target class. The same
            size applies to all target classes if an integer is provided.
            If a list is provided, ensure that the values correspond to the sorted target classes.
            Eg, for CMS types, it should be CMS1,CMS2,CMS3,CMS4.


        Returns:
            Tuple: Tuple containing a dataframe of only the newly generated samples and
            a second dataframe of the associated target labels.
        """

        target_count = dict(Counter(y[target]))
        sorted_count = sorted(target_count.items(), key=lambda item: item[1])
        max_count = sorted_count[-1][1]

        df = X.join(y[target])
        ref_df = df.sort_values(by=target)
        target_ids_all = ref_df.index
        target_ids = {
            k: ref_df[ref_df[target] == k].index for k in np.unique(y[target])
        }
        target_list = list(target_ids.keys())
        sampling_idx = pd.DataFrame(columns=target_list)
        sampled_labels = pd.DataFrame(columns=[target])
        row_idx = []
        sampled_df_list = []

        if isinstance(self.class_size, int):
            target_size_dict = dict.fromkeys(target_list, self.class_size)
        elif isinstance(self.class_size, list):
            class_key_values = list(zip(target_list, self.class_size))
            target_size_dict = dict(class_key_values)
        elif isinstance(self.class_size, dict):
            target_size_dict = self.class_size

        # NOTE: doing this row wise per class-block, so we sample all indices for class1
        size_list = []  # - to keep track of class sizes for indexing in new augmented df
        for item in target_list:
            row_idx = []

            size = (
                max_count - target_count[item]
                if self.class_size is None
                else target_size_dict[item]
                - target_count[
                    item
                ]  # remove "- target_count[item]" if you want to sample specified class_size samples
            )
            size_list.append(size)
            if size == 0:
                continue

            for subitem in target_list:
                if item == subitem:
                    target_class = np.random.choice(target_ids[item], size=size)
                    row_idx.append(target_class.tolist())

                else:
                    fragment = np.random.choice(
                        np.setdiff1d(target_ids_all, target_ids[subitem]), size
                    )
                    row_idx.append(fragment.tolist())

            sampled_cms = pd.DataFrame(row_idx, index=target_list).T
            sampling_idx = pd.concat([sampling_idx, sampled_cms])
            new_labels = pd.DataFrame({target: [item] * size})
            sampled_labels = pd.concat([sampled_labels, new_labels])

        sampling_idx = sampling_idx.reset_index(drop=True)
        sampled_labels = sampled_labels.reset_index(drop=True)

        ref_sample_index = []

        for i, item in enumerate(target_list):
            j = 0
            assert sum(sampling_idx[item].isin(target_ids[item])) == size_list[i]
            if size_list[i] == 0:
                continue
            sample_indices = sampling_idx.loc[
                size_list[i] * j : (size_list[i] * (j + 1)) - 1, item
            ]
            if sample_indices.dtype == int:
                sample_indices = sample_indices.astype(str)

            sample_indices = sample_indices + "S"

            ref_sample_index.append(sample_indices)
            j += 1

        for k, v in self.target_signatures.items():
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

    def intraclass_sampling(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        target: str,
    ):
        target_count = dict(Counter(y[target]))
        sorted_count = sorted(target_count.items(), key=lambda item: item[1])
        # max_label_count = max(target_count.items(),key=lambda k: k[1])
        max_count = sorted_count[-1][1]

        sampled_df = pd.DataFrame()
        sampled_labels = pd.DataFrame()

        target_list = sorted(list(self.target_signatures.keys()))

        if isinstance(self.class_size, int):
            target_size_dict = dict.fromkeys(target_list, self.class_size)
        elif isinstance(self.class_size, list):
            class_key_values = list(zip(target_list, self.class_size))
            target_size_dict = dict(class_key_values)
        elif isinstance(self.class_size, dict):
            target_size_dict = self.class_size

        for k, v in sorted_count:
            if k in ["NOLBL", math.nan]:
                continue
            subset_idx = np.argwhere(y[target].values == k).flatten()
            subset = X.iloc[subset_idx, :]
            # size = max_count - v if self.class_size is None else target_size_dict[k] - v
            # size = max_count - v if self.class_size is None else self.class_size - v
            size = (
                max_count - target_count[k]
                if self.class_size is None
                else target_size_dict[k]
                - target_count[
                    k
                ]  # remove "- target_count[item]" if you want to sample specified class_size samples
            )
            sampling_idx = np.random.randint(
                0, len(subset), size=(size, len(target_list))
            )
            sampling_idx = pd.DataFrame(sampling_idx, columns=target_list)
            sampled_df_list = []

            ref_sample_index = subset.index[sampling_idx[k]]
            if ref_sample_index.dtype == int:
                ref_sample_index = ref_sample_index.astype(str)
            ref_sample_index = ref_sample_index + "S"

            for key, value in self.target_signatures.items():
                idx = subset.index[sampling_idx.loc[:, key]]
                sampled_df_list.append(subset.loc[idx, value].reset_index())

            new_samples = pd.concat(sampled_df_list, axis=1).drop(columns="index")
            new_samples.index = ref_sample_index
            new_samples.index.name = "id"
            new_samples.index = new_samples.index.astype(str) + new_samples.groupby(
                "id"
            ).cumcount().astype(str)

            new_samples = new_samples.loc[:, X.columns]
            new_labels = pd.DataFrame({target: [k] * size}, index=new_samples.index)

            sampled_df = pd.concat([sampled_df, new_samples])
            sampled_labels = pd.concat([sampled_labels, new_labels])

        return sampled_df, sampled_labels

    def sample(self, X: pd.DataFrame, y: pd.DataFrame, target: str):
        sampled_df, sampled_labels = self.get_samples[self.sampling_method](
            X, y, target
        )
        return sampled_df, sampled_labels
