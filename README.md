# TCM: An efficient lightweight MLP-based network with affine transformation for long-term time series forecasting
This repo is the official implementation of TCM
## ⚙️Environment Requirements
We recommend using `Python>=3.8`, `PyTorch>=2.0.1`, and `CUDA>=12.1`.

Please make sure you have installed conda. Then, our environment can be installed by:
```sh
conda create -n TCM python=3.8
source activate TCM 
pip install -r requirements.txt
```

## Datasets
1. Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) as provided by Autoformer.
```sh
mkdir dataset
```
2. Place them in the `./dataset` directory.

## Training
All scripts for training and evaluation are located in the `scripts/` directory.

### Example: Training TCM on Electricity Dataset
To train the TCM model on the Electricity dataset, you can run the following command:
```sh
sh scripts/electricity.sh
```

## Acknowledgement
We express our gratitude to the following GitHub repositories for providing valuable codebases or datasets:
- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

- [Autoformer](https://github.com/thuml/Autoformer)

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)

- [MTS-Mixers](https://github.com/plumprc/MTS-Mixers)

## Citation
If you find this work helpful, please consider citing it as follows:
```
@article{JIANG2025128960,
  title = {TCM: An efficient lightweight MLP-based network with affine transformation for long-term time series forecasting},
  journal = {Neurocomputing},
  volume = {617},
  pages = {128960},
  year = {2025},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2024.128960},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231224017314},
  author = {Hongwei Jiang and Dongsheng Liu and Xinyi Ding and Yaning Chen and Hongtao Li}
}
```
