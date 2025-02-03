import torch
import numpy as np
import itertools
from nltk.metrics.distance import edit_distance as ed
from torchmetrics import Metric, MetricCollection


SUM = 'sum'


def get_metrics() -> MetricCollection:
    """
    Creates and returns a collection of metrics for evaluating OCR model performance.

    Returns:
        MetricCollection: A collection of metrics, including string match and edit distance.
    """
    return MetricCollection(
        {
            'string_match': StringMatchMetric(),
            'edit_distance': EditDistanceMetric(),
        })


class StringMatchMetric(Metric):
    """
    Computes the fraction of predictions that exactly match the target strings.

    Attributes:
        correct (torch.Tensor): Total number of correct matches.
        total (torch.Tensor): Total number of samples.
    """

    def __init__(self):
        """
        Initializes the metric state variables.
        """
        super().__init__()
        self.add_state(
            'correct',
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx=SUM,
        )
        self.add_state(
            'total',
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx=SUM,
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the metric state with predictions and targets.

        Args:
            preds (torch.Tensor): Predicted logits of shape [time_steps, batch_size, num_classes].
            target (torch.Tensor): Ground truth labels of shape [batch_size, time_steps].
        """
        batch_size = torch.tensor(target.shape[0])

        metric = torch.tensor(string_match(preds, target))

        self.correct += metric * batch_size
        self.total += batch_size

    def compute(self) -> torch.Tensor:
        """
        Computes the final string match metric.

        Returns:
            torch.Tensor: Fraction of exact matches between predictions and targets.
        """
        return self.correct / self.total


class EditDistanceMetric(Metric):
    """
    Computes the average edit distance between predicted and target strings.

    Attributes:
        correct (torch.Tensor): Total accumulated edit distance.
        total (torch.Tensor): Total number of samples.
    """

    def __init__(self):
        """
        Initializes the metric state variables.
        """
        super().__init__()
        self.add_state(
            'correct',
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx=SUM,
        )
        self.add_state(
            'total',
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx=SUM,
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the metric state with predictions and targets.

        Args:
            preds (torch.Tensor): Predicted logits of shape [time_steps, batch_size, num_classes].
            target (torch.Tensor): Ground truth labels of shape [batch_size, time_steps].
        """
        batch_size = torch.tensor(target.shape[0])

        metric = torch.tensor(edit_distance(preds, target))

        self.correct += metric * batch_size
        self.total += batch_size

    def compute(self) -> torch.Tensor:
        """
        Computes the final edit distance metric.

        Returns:
            torch.Tensor: Average edit distance between predictions and targets.
        """
        return self.correct / self.total


def string_match(pred: torch.Tensor, gt_labels: torch.Tensor) -> float:
    """
    Computes the fraction of exact string matches between predictions and ground truth.

    Args:
        pred (torch.Tensor): Predicted logits of shape [time_steps, batch_size, num_classes].
        gt_labels (torch.Tensor): Ground truth labels of shape [batch_size, time_steps].

    Returns:
        float: Fraction of exact matches in the batch.
    """
    pred = pred.permute(1, 0, 2)
    pred = pred.detach().cpu()
    pred = torch.argmax(pred, dim=2).numpy()

    gt_labels = gt_labels.detach().cpu().numpy()

    valid = 0
    for j in range(pred.shape[0]):
        p3 = [k for k, g in itertools.groupby(pred[j])]
        p3 = [k for k in p3 if k > 0]
        t = [k for k in gt_labels[j] if k > 0]
        valid += float(np.array_equal(p3, t))

    return valid / pred.shape[0]


def edit_distance(pred: torch.Tensor, gt_labels: torch.Tensor) -> float:
    """
    Computes the average edit distance between predicted and ground truth strings.

    Args:
        pred (torch.Tensor): Predicted logits of shape [time_steps, batch_size, num_classes].
        gt_labels (torch.Tensor): Ground truth labels of shape [batch_size, time_steps].

    Returns:
        float: Average edit distance in the batch.
    """
    pred = pred.permute(1, 0, 2)
    pred = pred.detach().cpu()
    pred = torch.argmax(pred, dim=2).numpy()

    gt_labels = gt_labels.detach().cpu().numpy()

    dist = 0
    for j in range(pred.shape[0]):
        p3 = [k for k, g in itertools.groupby(pred[j])]
        p3 = [k for k in p3 if k > 0]
        t = [k for k in gt_labels[j] if k > 0]

        s_pred = ''.join(chr(x) for x in p3)
        s_true = ''.join(chr(x) for x in t)

        dist += ed(s_pred, s_true)

    return dist / pred.shape[0]
