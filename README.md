[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Signature Informed Sampling for Transcriptomic Data [:page_facing_up:](https://doi.org/10.1093/bioadv/vbaf124)

The application of machine learning methods to biomedical applications has seen many successes. However, working with transcriptomic data on supervised learning tasks is challenging due to its high dimensionality, low patient numbers and class imbalances. Machine learning models tend to overfit these data and do not generalise well on out-of-distribution samples. Data augmentation strategies help alleviate this by introducing synthetic data points and acting as regularisers. However, existing approaches are either computationally intensive, require population parametric estimates or generate insufficiently diverse samples. To address these challenges, we introduce two classes of phenotype-driven data augmentation approaches – signature-dependent and signature-independent. The signature-dependent methods assume the existence of distinct gene signatures describing some phenotype and are simple, non-parametric, and novel data augmentation methods. The signature-independent methods are a modification of the established Gamma-Poisson and Poisson sampling methods for gene expression data. As case studies, we apply our augmentation methods to transcriptomic data of colorectal and breast cancer. Through discriminative and generative experiments with external validation, we show that our methods improve patient stratification by 5−15% over other augmentation methods in their respective cases. The study additionally provides insights into the limited benefits of over-augmenting data.

## Data Availability
For reproducibility purposes, we provide the *standardised augmented datasets* and corresponding *standardised test datasets* [here](https://zenodo.org/records/8383203).

## Installation

Create a conda environment:

```console
conda env create -f conda.yml
```

Activate the environment:

```console
conda activate sigsample
```

Install:

```console
pip install .
```

### development

Install in editable mode for development:

```sh
pip install --user -e .
```

## Examples

For some examples on how to use `signature_sampling` see [here](./scripts/data_gen.py).
For experiments on MLP and VAE, see [here](./scripts/)

## Citation
If any part of this code is used, please cite our work:
```
@article{janakarajan2025phenotype,
    author = {Janakarajan, Nikita and Graziani, Mara and Martínez, María Rodríguez},
    title = {Phenotype Driven Data Augmentation Methods for Transcriptomic Data},
    journal = {Bioinformatics Advances},
    pages = {vbaf124},
    year = {2025},
    month = {05},
    issn = {2635-0041},
    doi = {10.1093/bioadv/vbaf124},
    url = {https://doi.org/10.1093/bioadv/vbaf124},
    eprint = {https://academic.oup.com/bioinformaticsadvances/advance-article-pdf/doi/10.1093/bioadv/vbaf124/63309620/vbaf124.pdf},
}
```
