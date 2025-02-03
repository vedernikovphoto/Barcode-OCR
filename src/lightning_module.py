import torch
import pytorch_lightning as pl

from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.utils.train_utils import load_object
from src.models import CRNN


class OCRModule(pl.LightningModule):
    """
    PyTorch Lightning module for OCR tasks.

    Attributes:
        _config (Config): Configuration object.
        _model (torch.nn.Module): OCR model instance.
        _losses (list): List of loss functions with weights.
        _train_metrics (torchmetrics.Metric): Metrics for training.
        _valid_metrics (torchmetrics.Metric): Metrics for validation.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the OCR module with the specified configuration.

        Args:
            config (Config): Configuration object containing model, optimizer, and scheduler settings.
        """
        super().__init__()
        self._config = config
        self._model = CRNN(**self._config.mdl_kwargs)
        self._losses = get_losses(self._config.losses)

        metrics = get_metrics()
        self._train_metrics = metrics.clone(prefix='train_')
        self._valid_metrics = metrics.clone(prefix='valid_')

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Model output.
        """
        return self._model(x)

    def get_core_model(self) -> torch.nn.Module:
        """
        Returns the core PyTorch model for tracing.
        """
        return self._model

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            dict: Dictionary containing optimizer and scheduler configurations.
        """
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Executes a single training step.

        Args:
            batch (tuple): Tuple containing images, targets, and target lengths.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        images, targets, target_lengths = batch
        log_probs = self(images)

        batch_size = images.size(0)
        time_steps = log_probs.size(0)
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.int32)
        loss_value = self._calculate_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            'train_',
        )
        self._train_metrics(log_probs, targets)

        return loss_value

    def validation_step(self, batch, batch_idx) -> None:
        """
        Executes a single validation step.

        Args:
            batch (tuple): Tuple containing images, targets, and target lengths.
            batch_idx (int): Index of the current batch.
        """
        images, targets, target_lengths = batch
        log_probs = self(images)
        batch_size = images.size(0)
        time_steps = log_probs.size(0)
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.int32)

        self._calculate_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            'valid_',
        )
        self._valid_metrics(log_probs, targets)

    def on_train_epoch_start(self) -> None:
        """
        Resets training metrics at the start of each epoch.
        """
        self._train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        """
        Logs training metrics at the end of each epoch.
        """
        self.log_dict(self._train_metrics.compute(), on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        """
        Resets validation metrics at the start of each epoch.
        """
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """
        Logs validation metrics at the end of each epoch.
        """
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def _calculate_loss(    # noqa: WPS211
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        """
        Calculates the total weighted loss and logs individual losses.

        Args:
            log_probs (torch.Tensor): Log probabilities from the model.
            targets (torch.Tensor): Ground truth labels.
            input_lengths (torch.Tensor): Lengths of the input sequences.
            target_lengths (torch.Tensor): Lengths of the target sequences.
            prefix (str): Prefix for logging metrics.

        Returns:
            torch.Tensor: Total computed loss.
        """
        total_loss = torch.tensor(0, device=log_probs.device, dtype=log_probs.dtype)
        for cur_loss in self._losses:
            loss = cur_loss.loss(
                log_probs=log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )
            total_loss += cur_loss.weight * loss
            self.log(f'{prefix}{cur_loss.name}_loss', loss.item())
        self.log(f'{prefix}total_loss', total_loss.item())
        return total_loss
