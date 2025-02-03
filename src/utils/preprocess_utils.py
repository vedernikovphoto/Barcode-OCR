import cv2
import logging
from typing import Any, Dict, List
from pathlib import Path


def parse_coordinates(x_str: str, y_str: str, w_str: str, h_str: str) -> Dict[str, int]:
    """
    Parses coordinate values from strings and returns them as integers.

    Args:
        x_str (str): The string representing the x-coordinate of the top-left corner.
        y_str (str): The string representing the y-coordinate of the top-left corner.
        w_str (str): The string representing the width of the detection.
        h_str (str): The string representing the height of the detection.

    Raises:
        ValueError: If any of the values cannot be converted to an int.

    Returns:
        Dict[str, int]: A dictionary containing the parsed coordinates.
    """
    x_from = int(x_str)
    y_from = int(y_str)
    width = int(w_str)
    height = int(h_str)
    return {
        'x_from': x_from,
        'y_from': y_from,
        'width': width,
        'height': height,
    }


def deduplicate_detections(detections: List) -> List:
    """
    Removes duplicate detections based on bounding box coordinates.

    Args:
        detections (list of dict): List of detections for an image.

    Returns:
        list of dict: Deduplicated list of detections.
    """
    unique = []
    seen = set()
    for det in detections:
        det_tuple = (
            det['x_from'],
            det['y_from'],
            det['width'],
            det['height'],
        )
        if det_tuple not in seen:
            seen.add(det_tuple)
            unique.append(det)
    return unique


def get_image_files(input_images_dir: str) -> List[str]:
    """Retrieve all image files from the directory."""
    return [
        f.name for f in Path(input_images_dir).iterdir()
        if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    ]


def crop_and_resize_image(image_path: str, detections: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Crops and optionally resizes detected regions in an image.

    Args:
        image_path (str): Path to the original image.
        detections (list of dict): Detection results for the image.
        output_dir (str): Directory to save cropped images.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f'Failed to read image {image_path}')
        return

    image_h, image_w = image.shape[:2]

    for idx, det in enumerate(detections):
        x_from = det['x_from']
        y_from = det['y_from']
        width = det['width']
        height = det['height']

        x_min = max(0, x_from)
        y_min = max(0, y_from)
        x_max = min(image_w, x_min + width)
        y_max = min(image_h, y_min + height)

        cropped_img = image[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0:
            logging.debug(f'Empty crop for image {image_path}, detection {idx}')
            continue

        base_name = Path(image_path).stem
        output_file = Path(output_dir) / f'{base_name}.jpg'

        success = cv2.imwrite(output_file, cropped_img)
        if not success:
            logging.error(f'Failed to write image {output_file}')


def parse_line(line: str, line_number: int) -> Dict[str, object]:
    """
    Parses a single line from the TSV file and returns a dictionary with the parsed data.

    Args:
        line (str): A single line from the annotations TSV file.
        line_number (int): The line number in the TSV file, used for logging warnings.

    Returns:
        Dict[str, object]: If the line is valid, returns.
    """
    parts = line.strip().split('\t')
    if len(parts) < 6:
        logging.warning(f'Invalid format in line {line_number}: {line.strip()}')
        return {}

    filename = Path(parts[0]).name
    if filename.startswith('images/'):
        filename = filename[len('images/'):]

    code = parts[1]
    try:
        coords = parse_coordinates(
            x_str=parts[2],
            y_str=parts[3],
            w_str=parts[4],
            h_str=parts[5],
        )
    except ValueError as e:
        logging.warning(f'Error parsing line {line_number}: {e}')
        return {}

    detection = {'code': code}
    detection.update(coords)
    return {'filename': filename, 'detection': detection}


def read_annotations(annotations_file: str) -> Dict[str, List]:
    """
    Reads annotations from a TSV file.

    Args:
        annotations_file (str): Path to the annotations TSV file.

    Returns:
        dict: A mapping from image filenames to a list of detections.
    """
    detections_dict: Dict[str, List] = {}
    if not Path(annotations_file).is_file():
        logging.warning(f'Annotations file not found: {annotations_file}')
        return detections_dict

    with open(annotations_file, 'r', encoding='utf-8') as f:
        header = f.readline()   # noqa: F841
        for line_number, line in enumerate(f, start=2):
            parsed = parse_line(line, line_number)
            if not parsed:
                continue

            filename = parsed['filename']
            detection = parsed['detection']
            if filename not in detections_dict:
                detections_dict[filename] = []
            detections_dict[filename].append(detection)

    return detections_dict
