from typing import Optional

import lightning as pl
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import demo.logger
from demo.constants import Phase


local_logger = demo.logger.get_logger(__name__)


class ImageDataloader(pl.LightningDataModule):
    """Class to handle the image dataloader."""

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        transforms: dict[Phase, torchvision.transforms.Compose],
    ) -> None:
        """Initialize the ImageDataloader object."""

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.shuffle = shuffle

        self.trainset_dir: Optional[str] = None
        self.valset_dir: Optional[str] = None
        self.test_dir: Optional[str] = None

        local_logger.info("ImageDataloader (PyTorch Lightning) initialized.")

    def setup_for_training(self, trainset_dir: str, valset_dir: str, test_dir: Optional[str] = None) -> None:
        """Setup the dataloader.

        Args:
            trainset_dir (str): The path to the training dataset.
            valset_dir (str): The path to the validation dataset.
            test_dir (Optional[str]): The path to the test dataset.
        """

        self.trainset_dir = trainset_dir
        self.valset_dir = valset_dir
        self.test_dir = test_dir

        local_logger.info("ImageDataloader (PyTorch Lightning) setup for training.")

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader."""

        if not self.trainset_dir:
            raise ValueError("trainset_dir is not set.")

        return DataLoader(
            dataset=datasets.ImageFolder(
                root=self.trainset_dir,
                transform=self.transforms[Phase.TRAINING],
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader."""

        if not self.valset_dir:
            raise ValueError("valset_dir is not set.")

        return DataLoader(
            dataset=datasets.ImageFolder(
                root=self.valset_dir,
                transform=self.transforms[Phase.VALIDATION],
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get the test dataloader."""

        if not self.test_dir:
            local_logger.info("testset_dir is not set. Return None.")
            return None

        return DataLoader(
            dataset=datasets.ImageFolder(
                root=self.test_dir,
                transform=self.transforms[Phase.TESTING],
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
