from typing import Any, Callable, Optional

import lightning
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix, precision_score, roc_curve
from torch import nn

import demo.logger


local_logger = demo.logger.get_logger(__name__)


class ResNet18(lightning.LightningModule):
    """ResNet18 model using PyTorch Lightning."""

    def __init__(
        self,
        num_classes: int,
        class_labels: tuple[str, ...],
        denorm_fn: Optional[Callable] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize ResNet18 model.

        Args:
            num_classes (int): Number of classes.
            class_labels (tuple[str, ...]): Tuple of class labels.
            denorm_fn (Optional[Callable], optional): Function to denormalize the input.
                Defaults to None.
            hyperparameters (Optional[dict[str, Any]], optional): Hyperparameters for the model.
                Defaults to None.
        """

        super().__init__()

        self.hyparams = hyperparameters
        self.save_hyperparameters(self.hyparams)

        self.denorm_fn = denorm_fn
        self.class_labels = class_labels
        self.resnet = torchvision.models.resnet18(weights="DEFAULT")
        # Replace the final layer to match the number of classes
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

        # For roc curve and confusion matrix
        self._y_val_true = torch.tensor([])
        self._y_val_pred = torch.tensor([])
        self._loss_fn = nn.CrossEntropyLoss()

        self.best_precision = 0.0

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""

        optimizer = torch.optim.AdamW(self.parameters(), **self.hyparams["optimizer"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.hyparams["scheduler"])

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
            self.logger.experiment.add_image("sample_images_train", grid, self.current_epoch)

        logits = self.forward(x)
        loss = self._loss_fn(logits, y)
        self.log(name="train_loss", value=loss, on_step=True)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(name="train_acc", value=acc, on_step=True)

        self.log(name="lr", value=self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)

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
            self.logger.experiment.add_image("sample_images_val", grid, self.current_epoch)

        logits = self.forward(x)
        loss = self._loss_fn(logits, y)
        self.log(name="val_loss", value=loss, on_step=True)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(name="val_acc", value=acc, on_step=True)

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

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        self.best_precision = max(self.best_precision, precision)
        self.log(name="hp_metric", value=precision, on_epoch=True)

        self._y_val_true = torch.tensor([])
        self._y_val_pred = torch.tensor([])

    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, class_idx: int) -> None:
        """Plot ROC curve"""

        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, label=f"AUC = {np.trapz(tpr, fpr):.2f}")
        plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for Class {class_idx} ({self.class_labels[class_idx]})")
        plt.legend(loc="lower right")

        self.logger.experiment.add_figure(
            f"ROC Curve Class {class_idx} ({self.class_labels[class_idx]})",
            plt.gcf(),
            self.current_epoch,
        )
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix"""

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_true)))
        plt.yticks(tick_marks, self.class_labels)
        plt.ylabel("True Label")
        plt.xticks(tick_marks, self.class_labels, rotation=45)
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

        # Log to TensorBoard
        self.logger.experiment.add_figure("Confusion Matrix", plt.gcf(), self.current_epoch)
        plt.close()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""

        return self.resnet(x)
