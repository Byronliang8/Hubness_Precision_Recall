# Hubness Precision Recall For Generative Image
project page: https://byronliang8.github.io/Hubness_Precision_Recall_page/

This is the code for [paper](https://proceedings.mlr.press/v235/liang24f.html) and the hubness funcation CUDA version can used in [HubnessGANSampling](https://github.com/Byronliang8/HubnessGANSampling).

## Requirements 
- Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
- 64-bit Python 3.7 and PyTorch 1.7.1. See https://pytorch.org/ for PyTorch install instructions.
- CUDA toolkit 11.0 or later. Use at least version 11.1 if running on RTX 3090.
- Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3` and `pip install scikit-hubness`. 

## Usage
This code is based on [Github](https://github.com/youngjung/improved-precision-and-recall-metric-pytorch).

1) get the image features:
```
python getFeature.py datasetPath(real or fake images) --fname_precalc outputName.npz # The putput name should with .npz
```
2) compute precision and recall:
```
python improved_precision_recall_hubness.py real.npz fake.npz
```

## Citation
```
@InProceedings{pmlr-v235-liang24f,
  title = 	 {Efficient Precision and Recall Metrics for Assessing Generative Models using Hubness-aware Sampling},
  author =       {Liang, Yuanbang and Wu, Jing and Lai, Yu-Kun and Qin, Yipeng},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {29682--29699},
  year = 	 {2024},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
}
```
