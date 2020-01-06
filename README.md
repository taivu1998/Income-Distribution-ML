# Income-Distribution-ML

This program implements several machine learning and deep learning algorithms for prediction income distribution in California based on satellite imagery.

## Installation

The project requires the following frameworks:

- PyTorch: https://pytorch.org

- Scikit-learn: https://scikit-learn.org/stable/

- NumPy: https://numpy.org

- Pandas: https://pandas.pydata.org

## Usage

To run the program, use the following command:

```bash
python main.py
```

There are several optional command line arguments:

- --arch: ResNet architecture, such as 'resnet20' or 'resnet18'.
- --dataset: Dataset, such as 'cifar10' or 'cifar100'.
- --regularize: Regularization techniques, such as 'mixup' or 'cutout'.
- --prune: Pruning techniques, such as 'soft_filter'.
- --batch-size: Size of a training batch.
- --lr: Learning rate.
- --epochs: Number of training epochs.


## Authors

* **Tai Vu** - Stanford University
