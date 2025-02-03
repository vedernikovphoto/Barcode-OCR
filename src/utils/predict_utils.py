from typing import List, Tuple

import numpy as np
import torch
import itertools
import operator


def matrix_to_string(model_output: torch.Tensor, vocab: str) -> Tuple[List, List]:
    """
    Decodes a CTC matrix into strings and confidence scores.

    Args:
        model_output (torch.Tensor): Model output tensor of shape [time_steps, batch_size, num_classes].
        vocab (str): Vocabulary for decoding.

    Returns:
        Tuple[List[str], List[np.ndarray]]: Decoded strings and their confidence scores.
    """
    labels, confs = postprocess(model_output)
    labels_decoded, conf_decoded = decode(labels_raw=labels, conf_raw=confs)
    string_pred = labels_to_strings(labels_decoded, vocab)
    return string_pred, conf_decoded


def postprocess(model_output: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies softmax to the model output and extracts labels and confidences.

    Args:
        model_output (torch.Tensor): Model output tensor of shape [time_steps, batch_size, num_classes].

    Returns:
        Tuple[np.ndarray, np.ndarray]: Labels and their confidence scores.
    """
    output = model_output.permute(1, 0, 2)
    output = torch.nn.Softmax(dim=2)(output)
    confidences, labels = output.max(dim=2)
    confidences = confidences.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return labels, confidences


def decode(labels_raw: np.ndarray, conf_raw: np.ndarray) -> Tuple[List, List]:
    """
    Decodes raw labels and confidences by collapsing consecutive duplicates.

    Args:
        labels_raw (np.ndarray): Raw label array of shape [batch_size, time_steps].
        conf_raw (np.ndarray): Confidence array of shape [batch_size, time_steps].

    Returns:
        Tuple[List[List[int]], List[np.ndarray]]: Decoded labels and confidence scores.
    """
    result_labels = []
    result_confidences = []
    for label, conf in zip(labels_raw, conf_raw):
        result_one_labels = []
        result_one_confidences = []
        combined = zip(label, conf)  # Combine labels and confidences
        grouped = itertools.groupby(combined, key=operator.itemgetter(0))  # Group by label

        for current_label, group in grouped:
            if current_label > 0:
                result_one_labels.append(current_label)
                result_one_confidences.append(max(list(zip(*group))[1]))
        result_labels.append(result_one_labels)
        result_confidences.append(np.array(result_one_confidences))

    return result_labels, result_confidences


def labels_to_strings(labels: List[List[int]], vocab: str) -> List[str]:
    """
    Converts decoded labels into strings using a vocabulary.

    Args:
        labels (List[List[int]]): Decoded label sequences.
        vocab (str): Vocabulary for mapping label indices to characters.

    Returns:
        List[str]: Decoded strings.
    """
    strings = []
    for single_str_labels in labels:
        mapped_chars = []
        for char_index in single_str_labels:
            if char_index > 0:
                mapped_chars.append(vocab[char_index - 1])
            else:
                mapped_chars.append('_')
        try:    # noqa: WPS229
            output_str = ''.join(mapped_chars)
            strings.append(output_str)
        except IndexError:
            strings.append('Error')
    return strings
