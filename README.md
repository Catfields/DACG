
# DACG
DACG: Dual Attention and Context Guidance Model for Radiology Report Generation
# Overview

This repository contains code necessary to run DACG model.

## Requirements

- `torch==1.5.1`
- `torchvision==0.6.1`
- `opencv-python==4.4.0.42`

## Datasets
We use two datasets (IU X-Ray) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.



## Run on IU X-Ray

Run `run_iu_xray.sh` to train a model on the IU X-Ray data.
