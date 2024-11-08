from typing import Callable

import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm

import demo.logger


local_logger = demo.logger.get_logger(__name__)


def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Get an optimizer for the model.
    NOTE: here we just hard code the optimizer to be AdamW with lr=1e-3 for demonstration purposes.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        torch.optim.Optimizer: Optimizer.
    """

    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    local_logger.info("Optimizer initialized.")

    return optim


def get_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    """Get a scheduler for the optimizer.
    NOTE: here we just hard code the scheduler to be StepLR with step_size=1 and gamma=0.1 for demonstration purposes.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: Scheduler.
    """

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    local_logger.info("Scheduler initialized.")

    return scheduler


def train_regression_model(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: Callable,
    n_epochs: int,
    device: str = "cuda",
) -> None:
    """Train a regression model.

    Args:
        model (nn.Module): PyTorch model.
        train_dataloader (torch.utils.data.DataLoader): Training data loader.
        val_dataloader (torch.utils.data.DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler.
        n_epochs (int): Number of epochs.
        device (str, optional): Device to use.
            Defaults to "cuda".
    """

    model.to(device)

    for n in range(n_epochs):
        local_logger.info("Epoch %d / %d", n + 1, n_epochs)
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Train
        model.train()
        for x, y in tqdm(train_dataloader, desc="Training"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * x.size(0) / len(train_dataloader.dataset)

        # Here we step the scheduler after each epoch
        scheduler.step()

        local_logger.info("Training Loss: %.2f", epoch_train_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(val_dataloader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = nn.functional.mse_loss(y_pred, y)
                epoch_val_loss += loss.item() * x.size(0) / len(val_dataloader.dataset)

        local_logger.info("Validation Loss: %.2f", epoch_val_loss)
