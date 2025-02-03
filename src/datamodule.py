from typing import Optional
from pathlib import Path
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, RandomSampler

from src.constants import DATA_PATH
from src.config import Config
from src.transforms import get_transforms
from src.dataset import BarCodeDataset


class OCRDM(LightningDataModule):
    """
    Initialize the OCRDM data module.

    Args:
        config (Config): Configuration object containing parameters of the model.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._data_config = config.data_config
        self._augmentation_params = config.augmentation_params
        self._train_transforms = get_transforms(
            aug_config=self._augmentation_params,
            width=self._data_config.width,
            height=self._data_config.height,
            vocab=self._data_config.vocab,
            text_size=self._data_config.text_size,
        )
        self._valid_transforms = get_transforms(
            aug_config=self._augmentation_params,
            width=self._data_config.width,
            height=self._data_config.height,
            vocab=self._data_config.vocab,
            text_size=self._data_config.text_size,
            augmentations=False,
        )

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.train_sampler: Optional[RandomSampler] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training and validation.

        Args:
            stage (Optional[str]): Stage of the setup ('fit', 'test', etc.).
                                   If None, both training and validation datasets are initialized.
        """
        df_train = read_df(DATA_PATH, 'train')
        df_valid = read_df(DATA_PATH, 'valid')

        self.train_dataset = BarCodeDataset(
            df=df_train,
            data_folder=DATA_PATH,
            transforms=self._train_transforms,
        )
        self.valid_dataset = BarCodeDataset(
            df=df_valid,
            data_folder=DATA_PATH,
            transforms=self._valid_transforms,
        )

        if self._data_config.num_iterations != -1:
            self.train_sampler = RandomSampler(
                data_source=self.train_dataset,
                num_samples=self._data_config.num_iterations * self._data_config.batch_size,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader configured for training with batch size,
                        workers, and sampling options.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._data_config.batch_size,
            num_workers=self._data_config.n_workers,
            sampler=self.train_sampler,
            shuffle=not self.train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader configured for validation with batch size,
                        workers, and shuffle options.
        """
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._data_config.batch_size,
            num_workers=self._data_config.n_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    """
    Reads a DataFrame from a CSV file.

    Args:
        data_path (str): Path to the data directory.
        mode (str): Mode of the data ('train', 'valid').

    Returns:
        pd.DataFrame: DataFrame read from the TSV file.
    """
    file_name = f'df_{mode}.csv'
    file_path = Path(data_path) / file_name
    return pd.read_csv(file_path)
