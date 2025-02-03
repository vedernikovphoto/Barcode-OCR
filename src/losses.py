import typing as tp
from dataclasses import dataclass
from torch import nn
from src.config import LossConfig
from src.utils.train_utils import load_object


@dataclass
class Loss:
    """
    Represents a loss function with a name, weight, and the PyTorch loss module.

    Attributes:
        name (str): Name of the loss function.
        weight (float): Weight applied to the loss during training.
        loss (nn.Module): PyTorch loss function.
    """
    name: str
    weight: float
    loss: nn.Module


def get_losses(losses_cfg: tp.List[LossConfig]) -> tp.List[Loss]:
    """
    Initializes and returns a list of Loss objects from the configuration.

    Args:
        losses_cfg (tp.List[LossConfig]): List of loss configurations.

    Returns:
        tp.List[Loss]: List of initialized Loss objects.
    """
    return [
        Loss(
            name=loss_cfg.name,
            weight=loss_cfg.weight,
            loss=load_object(loss_cfg.loss_fn)(**loss_cfg.loss_kwargs),
        )
        for loss_cfg in losses_cfg
    ]
