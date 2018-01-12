# PGGAN-tensorflow
the Tensorflow implementation of [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](https://arxiv.org/abs/1710.10196).

### The generative process of PG-GAN

<p align="center">
  <img src="/images/figure.png">
</p>

## Differences with the original paper.

- This implement use CelebA dataset, not CelebA-HQ.

- All tricks have been used, except "Equalized learning rate". You can make a PR if high-qualtiy generated samples with this technique can be achieved. Thanks for your contributions.

- Recently, just generate 64x64 and 128x128 pixels samples.

## Setup

### Prerequisites

- TensorFlow >= 1.1
- python 2.7

### Getting Started
- Clone this repo:
```bash
git clone https://github.com/zhangqianhui/PGGAN-tensorflow
cd PGGAN-tensorflow
```
- Download the CelebA dataset

You can download the [CelebA dataset](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0) 
and unzip CelebA into a directory. Noted that this directory don't contain the sub-directory.

- Train the model
```bash
python main.py --path your data-path
```

## Results
Here is the generated 64x64 results of PGGAN-tensorflow(Left: generated; Right: Real):

<p align="center">
  <img src="/images/sample.png">
</p>

Here is the generated 128x128 results of PGGAN-tensorflow(Left: generated; Right: Real):
<p align="center">
  <img src="/images/sample_128.png">
</p>

## Issue
 If you find some bugs, Thanks for your issue to propose it.
    
## Reference code

[PGGAN Theano](https://github.com/tkarras/progressive_growing_of_gans)

[PGGAN Pytorch](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans)
