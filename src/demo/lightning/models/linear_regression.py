import lightning
import torch
from torch import nn

import demo.logger


local_logger = demo.logger.get_logger(__name__)


class LinearRegression(lightning.LightningModule):
    """Linear Regression Model."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the model."""

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

        local_logger.info("LinearRegression model (PyTorch Lightning) initialized.")

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass."""

        return self.linear(x)
