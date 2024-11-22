# Demo code for PyTorch/PyTorch Lightning

This README is a brief introduction to the code repository. It contains the following sections:
- [Installation: Dependencies for This Repository](#installation-dependencies-for-this-repository)
- [Demonstration: Descriptions of Notebooks](#demonstration-descriptions-of-notebooks)

## Installation: Dependencies for This Repository

```bash
# If you use poetry
poetry install

# If you don't use poetry
pip install -r requirements.txt .

# For development (not necessary)
pre-commit install
```

## Demonstration: Descriptions of Notebooks

- ### [01. Linear Regression](./notebooks/01_regression_model.ipynb)

    This notebook is for comparing how training is implemented in Native PyTorch and PyTorch Lightning. The aim of the notebook is to show what PyTorch Lightning does under the hood and how it simplifies the training process. The notebook contains the following sections:
    - Generate Data: Generate a simple dataset for training.
    - Native PyTorch Training: Implement training in Native PyTorch.
    - PyTorch Lightning Training: Implement training in PyTorch Lightning.

    In these sections, we will be mainly covering the following topics:
    - Syntax comparison between Native PyTorch and PyTorch Lightning.
    - Training loop implementation in Native PyTorch and PyTorch Lightning.

- ### [02. Resnet](./notebooks/02_resnet.ipynb)

    This notebook is a continuation of the previous notebook on PyTorch Lightning. In this notebook, we will explore more on the usage of PyTorch Lightning. The notebook contains the following sections:
    - Initialize Dataset: Initialize the CIFAR10 dataset for image classification task.
    - PyTorch Lightning: Train a ResNet.
    - PyTorch Lightning: Hyperparameter Search.
    - PyTorch Lightning: Built-in Callbacks.
    - PyTorch Lightning: Customized Callbacks.

    In these sections, we will be mainly covering the following topics:
    - Using torchvision with PyTorch Lightning
    - Training Monitoring (Visualization using TensorBoard)
    - Callbacks (Lightning built-in callbacks and customized callbacks)
