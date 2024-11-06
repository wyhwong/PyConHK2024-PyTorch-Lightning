import lightning
import torch
from torch import nn

import demo.logger
from demo.torch.utils import get_optimizer, get_scheduler


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

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""

        optimizer = get_optimizer(self)
        scheduler = get_scheduler(optimizer)

        return ([optimizer], [scheduler])

    def training_step(
        self,
        batch: torch.tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.tensor:
        """Training step."""

        x, y = batch
        y_pred = self.forward(x)
        loss = nn.functional.mse_loss(y_pred, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch: torch.tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.tensor:
        """Validation step."""

        x, y = batch
        y_pred = self.forward(x)
        loss = nn.functional.mse_loss(y_pred, y)

        self.log("val_loss", loss)
        return loss

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass."""

        return self.linear(x)
