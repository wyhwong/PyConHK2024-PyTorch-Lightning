from typing import Callable, Optional

import lightning
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix, roc_curve
from torch import nn

import demo.logger
from demo.torch.utils import get_optimizer, get_scheduler


local_logger = demo.logger.get_logger(__name__)


class ResNet(lightning.LightningModule):
    """ResNet model"""

    def __init__(self, num_classes: int, denorm_fn: Optional[Callable] = None) -> None:
        """Initialize the model"""

        super().__init__()

        self.denorm_fn = denorm_fn
        self.resnet = torchvision.models.resnet18(weights="default")
        # Replace the final layer to match the number of classes
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

        # For roc curve and confusion matrix
        self._y_val_true = torch.tensor([])
        self._y_val_pred = torch.tensor([])
        self._loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""

        optimizer = get_optimizer(self.resnet)
        scheduler = get_scheduler(optimizer)

        return ([optimizer], [scheduler])

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step"""

        x, y = batch

        if batch_idx == 0:
            x_denorm = self.denorm_fn(x) if self.denorm_fn else x
            grid = torchvision.utils.make_grid(x_denorm)
            # Here we ignore the type, expected message:
            # "Attribute 'experiment' is not defined for 'Optional[LightningLoggerBase]'"
            self.logger.experiment.add_image(  # type: ignore
                "sample_images_train",
                grid,
                self.current_epoch,
            )

        logits = self.forward(x)
        loss = self._loss_fn(logits, y)
        self.log(name="train_loss", value=loss, on_step=True)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(name="train_acc", value=acc, on_step=True)

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step"""

        x, y = batch

        if batch_idx == 0:
            x_denorm = self.denorm_fn(x) if self.denorm_fn else x
            grid = torchvision.utils.make_grid(x_denorm)
            # Here we ignore the type, expected message:
            # "Attribute 'experiment' is not defined for 'Optional[LightningLoggerBase]'"
            self.logger.experiment.add_image(  # type: ignore
                "sample_images_val",
                grid,
                self.current_epoch,
            )

        logits = self.forward(x)
        loss = self._loss_fn(logits, y)
        self.log(name="train_loss", value=loss, on_step=True)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(name="train_acc", value=acc, on_step=True)

        # For roc curve and confusion matrix
        probis = nn.functional.softmax(logits, dim=1)
        self._y_val_true = torch.cat([self._y_val_true.to(y.device), y])
        self._y_val_pred = torch.cat([self._y_val_pred.to(logits.device), probis])

        return loss

    def on_validation_epoch_end(self):
        """Validation epoch end"""

        for i in range(self._y_val_pred.shape[1]):  # Iterate over each class
            fpr, tpr, _ = roc_curve(
                y_true=(self._y_val_true == i).cpu().numpy(),
                y_score=self._y_val_pred[:, i].cpu().numpy(),
            )
            self.plot_roc_curve(fpr, tpr, i)

        y_true = self._y_val_true.cpu().numpy()
        y_pred = self._y_val_pred.argmax(dim=1).cpu().numpy()
        self.plot_confusion_matrix(y_true, y_pred)

        self._y_val_true = torch.tensor([])
        self._y_val_pred = torch.tensor([])

    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, class_index: int) -> None:
        """Plot the ROC curve

        Args:
            fpr (np.ndarray): The false positive rate
            tpr (np.ndarray): The true positive rate
            class_index (int): The class index
        """

        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, label=f"AUC = {np.trapz(tpr, fpr):.2f}")
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for Class {class_index}")
        plt.legend(loc="lower right")

        self.logger.experiment.add_figure(  # type: ignore
            f"ROC Curve Class {class_index} ({self.__model_config.backbone})",
            plt.gcf(),
            self.current_epoch,
        )
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot the confusion matrix

        Args:
            y_true (np.ndarray): The true labels
            y_pred (np.ndarray): The predicted labels
        """

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)  # type: ignore
        plt.title(f"Confusion Matrix ({self.__model_config.backbone})")
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_true)))
        plt.yticks(tick_marks, tick_marks)  # type: ignore
        plt.ylabel("True Label")
        plt.xticks(tick_marks, tick_marks)  # type: ignore
        plt.xlabel("Predicted Label")

        # Add text annotations
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        # Textual log
        local_logger.info("Confusion Matrix:\n%s", cm)

        # Log to TensorBoard
        self.logger.experiment.add_figure(  # type: ignore
            "Confusion Matrix",
            plt.gcf(),
            self.current_epoch,
        )
        plt.close()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""

        return self.resnet(x)
