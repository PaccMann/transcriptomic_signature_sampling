# signature_sampling
Signature Informed Sampling for Transcriptomic Data.


Transcriptomic data are challenging to work with in deep learning applications due its high dimensionality and low patient numbers. Deep learning models tend to overfit this data and not generalize well on out-of-distribution samples and new cohorts. Data augmentation strategies help alleviate this problem by introducing synthetic data points and acting as regularisers. However, the existing approaches are either computationally intensive or require parametric estimates. We introduce a new solution to an old problem - a simple, non-parametric, and novel data augmentation approach where gene signatures are crossed over between patients to generate new samples. As a case study, we apply our method to transcriptomic data of colorectal cancer. Through experiments on two different datasets, we show that our method improves patient stratification by generating samples that mirror biological variability and generalise to out-of-distribution data. Our approach requires little to no computation, and achieves performance on par with, if not better than, the existing augmentation methods.

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

For some examples on how to use `signature_sampling` see [here](./scripts/data_gen.py)
For experiments on MLP and VAE, see [here](./scripts/)
