# ECCV2020: Adaptive Mixture Regression Network with Local Counting Map for Crowd Counting


### Introduction

In this work, we introduce a new learning target named local counting map, and
show its feasibility and advantages in local counting regression. 

Paper: https://arxiv.org/abs/2005.05776


### Prerequisites
  - Python >= 3.5
  - Pytorch >= 1.0.1
  - other libs in ```requirements.txt```, run ```pip install -r requirements.txt```.

### Models
VGG16
QNRF-model (MAE/MSE: 86.6/152.1):

Google Drive: [download link](https://drive.google.com/open?id=1btZa7ltAwqQe0CDa41P67EtTdY0iJOfh),
Baidu Yun: [download link](https://pan.baidu.com/s/1humECw3oz4xRbWy5CaakZQ) (key: pe2r) 

### Run
- ```python crowdcounter.py```.
- results are saved at ```./images/results```.

### Based on
https://github.com/xiyang1012/Local-Crowd-Counting
