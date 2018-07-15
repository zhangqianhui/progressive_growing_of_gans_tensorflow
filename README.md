# PGGAN-tensorflow
the Tensorflow implementation of [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](https://arxiv.org/abs/1710.10196).

### The generative process of PG-GAN

<p align="center">
  <img src="/images/figure.png">
</p>

## Differences with the original paper.

- Recently, just generate 64x64 and 128x128 pixels samples.

## Setup

### Prerequisites

- TensorFlow >= 1.4
- python 2.7 or 3

### Getting Started
- Clone this repo:
```bash
git clone https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow.git
cd progressive_growing_of_gans_tensorflow
```
- Download the CelebA dataset

You can download the [CelebA dataset](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0) 
and unzip CelebA into a directory. Noted that this directory don't contain the sub-directory.

- The method for creating CelebA-HQ can be found on [Method](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans#how-to-create-celeba-hq-dataset)

- Train the model on CelebA dataset

```bash
python main.py --path=your celeba data-path --celeba=True
```

- Train the model on CelebA-HQ dataset

```bash
python main.py --path=your celeba-hq data-path --celeba=False
```

## Results on celebA dataset
Here is the generated 64x64 results(Left: generated; Right: Real):

<p align="center">
  <img src="/images/sample.png">
</p>

Here is the generated 128x128 results(Left: generated; Right: Real):
<p align="center">
  <img src="/images/sample_128.png">
</p>


## Results on CelebA-HQ dataset
Here is the generated 64x64 results(Left: Real; Right: Generated):

<p align="center">
  <img src="/images/hs_sample_64.jpg">
</p>

Here is the generated 128x128 results(Left: Real; Right: Generated):
<p align="center">
  <img src="/images/hs_sample_128.jpg">
</p>

## Issue
 If you find some bugs, Thanks for your issue to propose it.
    
## Reference code

[PGGAN Theano](https://github.com/tkarras/progressive_growing_of_gans)

[PGGAN Pytorch](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans)
