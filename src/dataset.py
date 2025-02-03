import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import pandas as pd
import albumentations as albu
import cv2
from torch.utils.data import Dataset


TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


class BarCodeDataset(Dataset):
    """
    Custom Dataset class for handling barcode image data and corresponding codes.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'filename' and 'code'.
            - 'filename': str, the file name of the image.
            - 'code': str, the text code corresponding to the barcode.

        data_folder (str): Path to the folder where the image files are stored.
        transforms (Optional[TRANSFORM_TYPE]): Optional albumentations transforms.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        data_folder: str,
        transforms: Optional[TRANSFORM_TYPE] = None,
    ):
        """
        Initialize the BarCodeDataset object.

        Args:
            df (pd.DataFrame): DataFrame with filenames and barcode codes.
            data_folder (str): Path to the folder containing the images.
            transforms (Optional[TRANSFORM_TYPE]): Optional data augmentation or preprocessing transforms.
        """
        self.transforms = transforms
        self.images = []
        self.codes = []

        for _, row in df.iterrows():
            file_name = row['filename']
            file_path = Path(data_folder) / file_name
            image = cv2.imread(str(file_path))
            image = image[..., ::-1]  # Convert BGR to RGB

            self.images.append(image)
            self.codes.append(str(row['code']))

    def __getitem__(self, idx) -> Tuple[np.ndarray, str, int]:
        """
        Retrieves an image and its corresponding barcode text and length.

        Args:
            idx (int): Index of the data item to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (numpy.ndarray): The transformed image.
                - text (str): The barcode text (code) associated with the image.
                - text_length (int): The length of the barcode text.

        Raises:
            IndexError: If the index is out of bounds.
        """
        text = self.codes[idx]
        image = self.images[idx]

        sample = {
            'image': image,
            'text': text,
            'text_length': len(text),
        }

        if self.transforms:
            sample = self.transforms(**sample)

        return sample['image'], sample['text'], sample['text_length']

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.images)
