import argparse
import logging
from typing import Any, Dict
from pathlib import Path
from src.utils.preprocess_utils import deduplicate_detections, crop_and_resize_image, get_image_files, read_annotations

logging.basicConfig(level=logging.INFO, format='{message}', style='{')


def arg_parse() -> Any:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Crop and resize images based on TSV annotations.')
    parser.add_argument(
        '--images-dir',
        type=str,
        default='data/original_images',
        help='Directory containing the original images.',
    )
    parser.add_argument(
        '--annotations-file',
        type=str,
        default='data/annotations.tsv',
        help='Path to the TSV file with annotations.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/images',
        help='Directory to save cropped and resized images.',
    )
    return parser.parse_args()


def process_single_image(image_file: str, input_images_dir: str, output_dir: str, detections_dict: Dict) -> int:
    """Process a single image and return the count of detections processed."""
    image_path = Path(input_images_dir) / image_file
    detections = detections_dict.get(image_file, [])

    if not detections:
        logging.debug(f'No detections for image {image_file}')
        return 0

    unique_detections = deduplicate_detections(detections)

    if not unique_detections:
        logging.debug(f'All detections are duplicates for image {image_file}')
        return 0

    logging.info(f'Processing {image_file} with {len(unique_detections)} unique detections')
    crop_and_resize_image(image_path, unique_detections, output_dir)
    return len(unique_detections)


def process_images(input_images_dir: str, output_dir: str, detections_dict: Dict) -> None:
    """
    Processes images by cropping them based on provided detections.

    Args:
        input_images_dir (str): Directory with original images.
        output_dir (str): Directory to save cropped images.
        detections_dict (dict): Dictionary mapping filenames to detections.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(input_images_dir)
    total_images = len(image_files)
    processed_images = 0
    total_detections = 0

    for image_file in image_files:
        detections_count = process_single_image(image_file, input_images_dir, output_dir, detections_dict)
        if detections_count > 0:
            processed_images += 1
            total_detections += detections_count

    logging.info(f'Total images found: {total_images}. Total images processed: {processed_images}')
    logging.info(f'Total detections processed: {total_detections}')


def main():
    args = arg_parse()
    detections_dict = read_annotations(args.annotations_file)
    process_images(args.images_dir, args.output_dir, detections_dict)


if __name__ == '__main__':
    main()
