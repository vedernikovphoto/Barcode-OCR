from typing import List
from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    """
    Configuration for a loss function.

    Attributes:
        name (str): The name of the loss function.
        weight (float): The weight of this loss in the total loss computation.
        loss_fn (str): The identifier for the loss function (e.g., module path).
        loss_kwargs (dict): Additional parameters for the loss function.
    """
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    """
    Configuration for data-related parameters.

    Attributes:
        batch_size (int): Number of samples per training batch.
        num_iterations (int): Number of iterations per epoch.
        n_workers (int): Number of worker threads for data loading.
        width (int): Input image width after resizing.
        height (int): Input image height after resizing.
        vocab (str): String containing all valid characters for the OCR.
        text_size (int): Size of text for rendering synthetic data.
    """
    batch_size: int
    num_iterations: int
    n_workers: int
    width: int
    height: int
    vocab: str
    text_size: int


class AugmentationConfig(BaseModel):
    """
    Configuration for data augmentation.

    Attributes:
        crop_perspective_p (float): Probability of applying a perspective crop transformation.
        scale_x_p (float): Probability of scaling the image along the x-axis.

        random_brightness_contrast_p (float): Probability of applying random brightness/contrast adjustments.
        clahe_p (float): Probability of applying CLAHE (Contrast Limited Adaptive Histogram Equalization).

        blur_p (float): Probability of applying a blur to the image.
        blur_limit (int): Maximum kernel size for the blur operation.

        gauss_noise_p (float): Probability of adding Gaussian noise to the image.

        downscale_p (float): Probability of downscaling the image to a lower resolution and then upsampling it.
        downscale_scale_min (float): Minimum scale factor for downscaling the image.
        downscale_scale_max (float): Maximum scale factor for downscaling the image.

        coarse_dropout_p (float): Probability of applying CoarseDropout to mask out random regions of the image.
        coarse_dropout_max_holes (int): Maximum number of regions (holes) to mask out with CoarseDropout.
        coarse_dropout_min_holes (int): Minimum number of regions (holes) to mask out with CoarseDropout.
    """
    crop_perspective_p: float
    scale_x_p: float
    random_brightness_contrast_p: float
    clahe_p: float
    blur_p: float
    blur_limit: int
    gauss_noise_p: float
    downscale_p: float
    downscale_scale_min: float
    downscale_scale_max: float
    coarse_dropout_p: float
    coarse_dropout_max_holes: int
    coarse_dropout_min_holes: int


class Config(BaseModel):
    """
    Main configuration class for the OCR project.

    Attributes:
        project_name (str): Name of the project.
        experiment_name (str): Identifier for the experiment.
        data_config (DataConfig): Configuration for the dataset.
        n_epochs (int): Number of epochs for training.
        log_every_n_steps (int): Frequency (in steps) for logging metrics.
        num_classes (int): Number of output classes for the model.
        accelerator (str): Type of hardware accelerator ('gpu' or 'cpu').
        seed (int): Random seed for reproducibility.
        device (int): Device index for the hardware accelerator.
        monitor_metric (str): Metric used for monitoring performance.
        monitor_mode (str): Mode for optimizing the monitored metric ('min' or 'max').
        mdl_kwargs (dict): Parameters specific to the model architecture.
        optimizer (str): Optimizer used for training.
        optimizer_kwargs (dict): Additional parameters for the optimizer.
        scheduler (str): Learning rate scheduler.
        scheduler_kwargs (dict): Additional parameters for the scheduler.
        losses (List[LossConfig]): List of loss function configurations.

    Methods:
        from_yaml(cls, path: str) -> 'Config': Load configuration from a YAML file.
    """
    project_name: str
    experiment_name: str
    data_config: DataConfig
    augmentation_params: AugmentationConfig
    n_epochs: int
    log_every_n_steps: int
    num_classes: int
    accelerator: str
    seed: int
    device: int
    monitor_metric: str
    monitor_mode: str
    mdl_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    losses: List[LossConfig]

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Load configuration from a YAML file.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            Config: Loaded configuration object.
        """
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
