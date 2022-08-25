# ETIC
A package to measure the dependence between two random elements given i.i.d. pairs of them with the entropy-regularized optimal transport independence criterion (ETIC), introduced in this [paper](http://arxiv.org/abs/2112.15265) in AISTATS 2022 (Oral presentation).

## Prerequisites
This package is based on [PyTorch](https://pytorch.org/). Other dependencies can be found in the file [environment.yml](environment.yml).
If using ``conda``, run the following command to install all required packages and activate the environment:
```bash
$ conda env create --file environment.yml
$ conda activate etic
```

## Installation
Clone the repository here:
```bash
$ git clone https://github.com/langliu95/etic.git
$ cd etic/
```


## Citation
If you find this package useful, or you use it in your research, please cite:

```
@inproceedings{lph,
title = {Entropy Regularized Optimal Transport Independence Criterion},
author = {Liu, Lang and
          Pal, Soumik and
          Harchaoui, Zaid},
booktitle = {AISTATS},
year = {2022}
}
```

## Acknowledgements
This work was supported by NSF CCF-2019844, NSF DMS-2023166, NSF DMS-2052239, a PIMS CRG (PIHOT), NSF DMS-2134012, CIFAR-LMB, and faculty research awards.
