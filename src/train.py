from typing import Any
from clearml import Task
import torch
import logging
import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar, ModelCheckpoint
from pytorch_lightning import seed_everything
from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import OCRDM
from src.lightning_module import OCRModule


logging.basicConfig(level=logging.INFO)


def arg_parse() -> Any:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def save_torchscript_model(model, save_path, example_input) -> torch.Tensor:
    """
    Converts the PyTorch model to TorchScript and saves it.

    Args:
        model (nn.Module): The model to be saved.
        save_path (str): The path where to save the TorchScript model.
        example_input (torch.Tensor): Example input for tracing the model.
    """
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, save_path)     # noqa: S614


def train(config: Config) -> None:
    """
    Trains and evaluates the OCR model.

    Sets up the data module, model, ClearML task, and PyTorch Lightning Trainer
    with necessary callbacks for training and evaluation.

    Args:
        config (Config): Configuration object for training.

    Returns:
        None
    """
    datamodule = OCRDM(config)
    model = OCRModule(config)

    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True,
    )
    task.connect(config.dict())

    experiment_save_path = Path(EXPERIMENTS_PATH) / config.experiment_name
    experiment_save_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            RichProgressBar(),
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)

    # Loading the best model from checkpoint. Extracting the core model from the loaded checkpoint
    best_model = OCRModule.load_from_checkpoint(checkpoint_callback.best_model_path, config=config)
    core_model = best_model.get_core_model()

    # Moving core model to device. Setting core model to evaluation mode
    core_model.to(config.device)
    core_model.eval()

    # Generating example input for model tracing. Saving the TorchScript model
    height = config.data_config.height
    width = config.data_config.width
    example_input = torch.randn(1, 3, height, width).to(config.device)
    torchscript_model_path = Path(experiment_save_path) / 'model.pt'
    save_torchscript_model(core_model, torchscript_model_path, example_input)

    logging.info(f'TorchScript model saved successfully at {torchscript_model_path}')


if __name__ == '__main__':
    args = arg_parse()
    config = Config.from_yaml(args.config_file)
    seed_everything(config.seed, workers=True)
    train(config)
