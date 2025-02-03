from typing import Union, List, Dict
import cv2
import numpy as np
from numpy import random

import albumentations as albu
import torch
from albumentations.pytorch import ToTensorV2
from src.config import AugmentationConfig


TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]
RANDOM = 'random'
LEFT = 'left'
IMAGE = 'image'


def get_transforms(     # noqa: WPS211
    aug_config: AugmentationConfig,
    width: int,
    height: int,
    text_size: int,
    vocab: Union[str, List[str]],
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> TRANSFORM_TYPE:
    """
    Builds a transformation pipeline for OCR preprocessing, augmentation, and postprocessing.

    Args:
        width (int): Target image width.
        height (int): Target image height.
        text_size (int): Maximum length of encoded text.
        vocab (Union[str, List[str]]): Vocabulary for text encoding.
        preprocessing (bool): If True, apply preprocessing transformations.
        augmentations (bool): If True, apply data augmentations.
        postprocessing (bool): If True, apply postprocessing transformations.

    Returns:
        TRANSFORM_TYPE: Composed albumentations transformation pipeline.
    """

    transforms = []

    if augmentations:
        transforms.extend([
            CropPerspective(p=aug_config.crop_perspective_p),
            ScaleX(p=aug_config.scale_x_p),
        ])

    if preprocessing:
        transforms.append(
            PadResizeOCR(
                target_height=height,
                target_width=width,
                mode=RANDOM if augmentations else LEFT,
            ))

    if augmentations:
        transforms.extend([
            albu.RandomBrightnessContrast(p=aug_config.random_brightness_contrast_p),
            albu.CLAHE(p=aug_config.clahe_p),
            albu.Blur(blur_limit=aug_config.blur_limit, p=aug_config.blur_p),
            albu.GaussNoise(p=aug_config.gauss_noise_p),
            albu.Downscale(
                scale_min=aug_config.downscale_scale_min,
                scale_max=aug_config.downscale_scale_max,
                p=aug_config.downscale_p,
            ),
            albu.CoarseDropout(
                max_holes=aug_config.coarse_dropout_max_holes,
                min_holes=aug_config.coarse_dropout_min_holes,
                p=aug_config.coarse_dropout_p,
            ),
        ])

    if postprocessing:
        transforms.extend([
            albu.Normalize(),
            TextEncode(vocab=vocab, target_text_size=text_size),
            ToTensorV2(),
        ])

    return albu.Compose(transforms)


class PadResizeOCR:
    """
    Resizes an image to the target size while preserving aspect ratio, adding padding if needed.

    Args:
        target_width (int): Target width.
        target_height (int): Target height.
        padding (int): Padding value (default: 0).
        mode (str): Padding mode (RANDOM, LEFT, 'center').

    Raises:
        ValueError: If mode is not one of the supported values.
    """

    def __init__(self, target_width, target_height, padding: int = 0, mode: str = RANDOM):
        self.target_width = target_width
        self.target_height = target_height
        self.padding = padding
        self.mode = mode

        if self.mode not in {RANDOM, LEFT, 'center'}:
            raise ValueError(f"Invalid mode. Expected one of {RANDOM, LEFT, 'center'}.")

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        """
        Resizes and pads the input image.

        Args:
            kwargs (dict): Dictionary containing the IMAGE key.

        Returns:
            Dict[str, np.ndarray]: Dictionary with the resized and padded image.
        """
        image = kwargs[IMAGE].copy()
        h, w = image.shape[:2]
        tmp_w = min(int(w * (self.target_height / h)), self.target_width)
        image = cv2.resize(image, (tmp_w, self.target_height))

        dw = np.round(self.target_width - tmp_w).astype(int)
        if dw > 0:
            if self.mode == RANDOM:
                pad_left = np.random.randint(dw)
            elif self.mode == LEFT:
                pad_left = 0
            else:
                pad_left = dw // 2

            pad_right = dw - pad_left

            image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        kwargs[IMAGE] = image
        return kwargs


class TextEncode:
    """
    Encodes text into numerical indices using a vocabulary.

    Args:
        vocab (Union[str, List[str]]): Vocabulary for encoding.
        target_text_size (int): Maximum length of encoded text.
    """

    def __init__(self, vocab: Union[str, List[str]], target_text_size: int):
        self.vocab = vocab if isinstance(vocab, list) else list(vocab)
        self.target_text_size = target_text_size

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        """
        Encodes the text in the input dictionary.

        Args:
            kwargs (dict): Dictionary containing the 'text' key.

        Returns:
            Dict[str, np.ndarray]: Dictionary with encoded text.
        """
        source_text = kwargs['text'].strip()
        postprocessed_text = []
        for x in source_text:
            if x in self.vocab:
                postprocessed_text.append(self.vocab.index(x) + 1)

        postprocessed_text = np.pad(
            postprocessed_text,
            pad_width=(0, self.target_text_size - len(postprocessed_text)),
            mode='constant',
        )
        postprocessed_text = torch.IntTensor(postprocessed_text)

        kwargs['text'] = postprocessed_text

        return kwargs


class CropPerspective:
    """
    Applies a random perspective transformation to the image.

    Args:
        p (float): Probability of applying the transformation.
        width_ratio (float): Horizontal distortion range as a ratio of width.
        height_ratio (float): Vertical distortion range as a ratio of height.
    """
    def __init__(self, p: float = 0.5, width_ratio: float = 0.04, height_ratio: float = 0.08):
        self.p = p
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        """
        Applies the perspective transformation if the probability condition is met.

        Args:
            kwargs (dict): Dictionary containing the IMAGE key.

        Returns:
            Dict[str, np.ndarray]: Dictionary with the transformed image.
        """
        image = kwargs[IMAGE].copy()

        if random.random() < self.p:
            h, w, c = image.shape

            pts1 = np.float32([[0, 0], [0, h], [w, h], [w, 0]])     # noqa: WPS221
            dw = w * self.width_ratio
            dh = h * self.height_ratio

            pts2 = np.float32(
                [
                    [random.uniform(-dw, dw), random.uniform(-dh, dh)],
                    [random.uniform(-dw, dw), h - random.uniform(-dh, dh)],       # noqa: WPS221
                    [w - random.uniform(-dw, dw), h - random.uniform(-dh, dh)],   # noqa: WPS221
                    [w - random.uniform(-dw, dw), random.uniform(-dh, dh)],       # noqa: WPS221
                ])

            matrix = cv2.getPerspectiveTransform(pts2, pts1)
            dst_w = (pts2[3][0] + pts2[2][0] - pts2[1][0] - pts2[0][0]) * 0.5     # noqa: WPS221
            dst_h = (pts2[2][1] + pts2[1][1] - pts2[3][1] - pts2[0][1]) * 0.5     # noqa: WPS221
            image = cv2.warpPerspective(
                src=image,
                M=matrix,
                dsize=(int(dst_w), int(dst_h)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        kwargs[IMAGE] = image
        return kwargs


class ScaleX:
    """
    Applies random scaling along the width axis.

    Args:
        p (float): Probability of applying the transformation.
        scale_min (float): Minimum scaling factor.
        scale_max (float): Maximum scaling factor.
    """
    def __init__(self, p: float = 0.5, scale_min: float = 0.8, scale_max: float = 1.2):
        self.p = p
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        """
        Applies the scaling transformation if the probability condition is met.

        Args:
            kwargs (dict): Dictionary containing the IMAGE key.

        Returns:
            Dict[str, np.ndarray]: Dictionary with the scaled image.
        """
        image = kwargs[IMAGE].copy()

        if random.random() < self.p:
            h, w, c = image.shape
            w = int(w * random.uniform(self.scale_min, self.scale_max))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        kwargs[IMAGE] = image
        return kwargs
