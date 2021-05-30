# ECCV2020: Adaptive Mixture Regression Network with Local Counting Map for Crowd Counting


### Introdução

Neste trabalho, apresentamos um modelo para contagem de mutidão, utilizando 
Paper: https://arxiv.org/abs/2005.05776


### Prerequisitos
  - Python >= 3.5
  - Pytorch >= 1.0.1
  - outras libs em ```requirements.txt```, rode ```pip install -r requirements.txt```.

### Modelos
VGG16
QNRF-model (MAE/MSE: 86.6/152.1):

Google Drive: [download link](https://drive.google.com/open?id=1btZa7ltAwqQe0CDa41P67EtTdY0iJOfh),
Baidu Yun: [download link](https://pan.baidu.com/s/1humECw3oz4xRbWy5CaakZQ) (key: pe2r) 

### Rodar
- ```python crowdcounter.py```.
- results are saved at ```./images/results```.

### Baseado em
https://github.com/xiyang1012/Local-Crowd-Counting
Paper: https://arxiv.org/abs/2005.05776

### Primeira etapa de desenvolvimento
Redução de todo o código para uma única classe capaz de fazer a contagem utilizando o melhor modelo proposto pelo paper (crowdcounter.py).

### Segunda etapa de desenvolvimento
Criação de um mapa de contagem com melhor visualização de onde exise maior concentração de pessoas.
