import torch
from timm import create_model


class CRNN(torch.nn.Module):
    """
    Implements a CRNN (Convolutional Recurrent Neural Network) for OCR tasks.

    Combines a CNN backbone for feature extraction with a GRU-based RNN
    for sequential modeling and a final classifier.

    Args:
        backbone_name (str): Name of the CNN backbone from timm.
        pretrained (bool): Whether to use a pretrained backbone.
        cnn_output_size (int): Output channel size from the CNN backbone.
        rnn_features_num (int): Number of features to pass to the RNN.
        rnn_hidden_size (int): Hidden size of the GRU.
        rnn_dropout (float): Dropout rate in the GRU.
        rnn_bidirectional (bool): Whether the GRU is bidirectional.
        rnn_num_layers (int): Number of GRU layers.
        num_classes (int): Number of output classes for classification.
    """

    def __init__(   # noqa: WPS211
        self,
        backbone_name: str = 'resnet18',
        pretrained: bool = True,
        cnn_output_size: int = 128,
        rnn_features_num: int = 48,
        rnn_hidden_size: int = 64,
        rnn_dropout: float = 0.1,
        rnn_bidirectional: bool = True,
        rnn_num_layers: int = 2,
        num_classes: int = 11,
    ) -> None:
        super().__init__()

        # CNN backbone for feature extraction
        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2,),
        )

        # Convolutional layer to adjust CNN output dimensions
        self.gate = torch.nn.Conv2d(cnn_output_size, rnn_features_num, kernel_size=1, bias=False)

        # GRU-based RNN for sequence modeling
        self.rnn = torch.nn.GRU(
            input_size=576,     # noqa: WPS432
            hidden_size=rnn_hidden_size,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
            num_layers=rnn_num_layers,
        )

        # Fully connected classifier
        classifier_in_features = rnn_hidden_size
        if rnn_bidirectional:
            classifier_in_features = 2 * rnn_hidden_size
        self.fc = torch.nn.Linear(classifier_in_features, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CRNN.

        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Log-softmax probabilities of shape [time_steps, batch_size, num_classes].
        """
        # Extract CNN features
        cnn_features = self.backbone(tensor)[0]
        cnn_features = self.gate(cnn_features)

        # Prepare for RNN
        cnn_features = cnn_features.permute(3, 0, 2, 1)
        cnn_features = cnn_features.reshape(
            cnn_features.shape[0],
            cnn_features.shape[1],
            cnn_features.shape[2] * cnn_features.shape[3],
        )

        # Pass through RNN
        rnn_output, _ = self.rnn(cnn_features)

        # Classification
        logits = self.fc(rnn_output)
        return self.softmax(logits)
