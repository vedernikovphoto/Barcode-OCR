import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.transforms import get_transforms
from src.utils.predict_utils import matrix_to_string
from src.config import Config


@torch.inference_mode()
def run_inference(
    model,
    image_path: Path,
    device: torch.device,
    transforms,
    vocab: str,
) -> str:
    """
    Runs inference on a single image using the given OCR model.

    Args:
        model: The trained OCR model (either an OCRModule or a TorchScript model).
        image_path (Path): Path to the image file.
        device (torch.device): Computation device.
        transforms: Preprocessing transforms.
        vocab (str): String of possible characters.

    Returns:
        str: The predicted text from the image.
    """

    image = Image.open(image_path).convert('RGB')

    transformed_image = transforms(image=np.array(image), text='')['image']
    transformed_image = transformed_image.unsqueeze(0).to(device)

    output = model(transformed_image).cpu().detach()

    # Convert output matrix to string
    string_pred, _ = matrix_to_string(output, vocab)
    return string_pred[0] if string_pred else ''


def parse_args() -> Any:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Inference script for OCR model')
    parser.add_argument(
        '--model-checkpoint',
        type=str,
        default='model_weights/ocr_model_weights.pt',
        help='Path to the model file',
    )
    parser.add_argument('--config-file', type=str, default='configs/config.yaml', help='Path to the config file')
    parser.add_argument('--input-folder', type=str, default='inference_images', help='Path to input images folder')
    parser.add_argument('--output', type=str, default='ocr_predictions.csv', help='Path to CSV file for predictions')
    return parser.parse_args()


def load_model(model_path: str, device: torch.device):
    """
    Loads the model from a .pt file.

    Args:
        model_path (str): Path to the model file.
        config (Config): Configuration object.
        device (torch.device): Computation device.

    Returns:
        The loaded model.
    """
    model = torch.jit.load(model_path, map_location=device)     # noqa: S614
    model.eval()
    return model


def main():
    args = parse_args()
    config = Config.from_yaml(args.config_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define vocabulary and transforms
    vocab = config.data_config.vocab
    transforms = get_transforms(
        aug_config=config.augmentation_params,
        width=config.data_config.width,
        height=config.data_config.height,
        text_size=config.data_config.text_size,
        vocab=vocab,
        postprocessing=True,
        augmentations=False,
    )

    # Load model and images for inference
    model = load_model(args.model_checkpoint, device)
    input_folder = Path(args.input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        raise FileNotFoundError(f'Input folder not found: {input_folder}')

    image_paths = []
    for p in input_folder.iterdir():
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            image_paths.append(p)

    if not image_paths:
        raise FileNotFoundError('No valid images found in the input folder.')

    predictions = []
    for image_path in tqdm(image_paths, desc='Running OCR Inference'):
        pred_text = run_inference(model, image_path, device, transforms, vocab)
        predictions.append({'image_name': image_path.name, 'predicted_text': pred_text})

    df = pd.DataFrame(predictions)
    df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
