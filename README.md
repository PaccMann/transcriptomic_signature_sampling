[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Signature Informed Sampling for Transcriptomic Data


Transcriptomic data are challenging to work with in deep learning applications due their high dimensionality and low patient numbers. Deep learning models tend to overfit this data, and do not generalize well on out-of-distribution samples and new cohorts. Data augmentation strategies help alleviate this problem by introducing synthetic data points and acting as regularisers. However, the existing approaches are either computationally intensive or require parametric estimates. We introduce a new solution to an old problem - a simple, non-parametric, and novel data augmentation approach where gene signatures are crossed over between patients to generate new samples. As a case study, we apply our method to transcriptomic data of colorectal cancer. Through experiments on two different datasets, we show that our method improves patient stratification by generating samples that mirror biological variability and generalise to out-of-distribution data. Our approach requires little to no computation, and achieves performance on par with, if not better than, the existing augmentation methods.

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
