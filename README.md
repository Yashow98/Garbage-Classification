# CNN for Garbage classification
### Introduction

This repository is built for garbage classification task, which contains full code and dataset. 

### Usage

1. Requirement:

   - Hardware: tested with GEFORCE RTX4060(8G)
   - Software: tested with PyTorch 1.12.1, Python3.9, CUDA 11.6

2. Clone the repository:

   ```shell
   git clone https://github.com/Yashow98/Garbage-classification.git
   ```
   
3. Train:

   ```
   
   ```

### Performance

Overall result:

|   Method    | Acc  | mF1-score | Params  | Flops(MACs) |
| :---------: | :--: | :-------: | :-----: | :---------: |
|   alexnet   |      |           | 14.59M  |   310.07M   |
|    vgg11    |      |           | 128.79M |    7.63G    |
|    vgg13    |      |           | 128.98M |   11.34G    |
|    vgg16    |      |           | 134.29M |    15.5G    |
|    vgg19    |      |           | 139.59M |   19.66G    |
|  resnet18   |      |           | 11.18M  |    1.82G    |
|  resnet34   |      |           | 21.29M  |    3.68G    |
|  resnet50   |      |           | 23.52M  |    4.12G    |
|  resnet101  |      |           | 42.51M  |    7.85G    |
|  resnet152  |      |           | 58.16M  |   11.58G    |
| Densenet121 |      |           |  6.96M  |    2.88G    |
|             |      |           |         |             |
|             |      |           |         |             |
|             |      |           |         |             |

