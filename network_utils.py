import torch
import torch.nn as nn
from typing import Any, Mapping

class SequenceEncoder(nn.Module):
    """
        Takes sequence of OMNI parameters as input and generates aggregated
        intermediate features.
    """
    default_search_space: Mapping[str, Any]

    def __init__(self, input_dim: int, output_dim: int):
        """
            :param input_dim: number of input features
            :param output_dim: number of output features
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, omni_sequence: torch.Tensor) -> torch.Tensor:
        """
            omni_sequence: shape: (batch_size, sequence_length, input_dim)
            output: shape: (batch_size, output_dim)
        """
        raise NotImplementedError


def activation_factory(activation_type):
    """
    This function creates pytorch's nn.functional object of specified
    activation
    :param input_x: input vector
    :param activationType:
    :return: torch.nn.functional acivation
    """
    localizers = {
        "relu": nn.ReLU(),
        "leakyRelu": nn.LeakyReLU(),
        "pRelu": nn.PReLU(),
    }

    return localizers[activation_type]


class LSTM(SequenceEncoder):
    """
        Takes sequence of OMNI parameters as input and generates aggregated
        intermediate features.
    """

    default_search_space = {
        "input_dim": 12,
        "hidden_layer_size": 128,
        "output_dim_encoder": 52,
        "layers": 2
    }

    def __init__(self, input_dim: int = 12, hidden_layer_size: int = 128,
                 output_dim_encoder: int = 12, layers: int = 2, layer_norm: bool = False):
        """
            :param input_dim: Number of input features for each data point in
            the window.
            :param hidden_layer_size: Number of variables the hidden layers.
            :param output_dim: Number of aggregated transformed output features.
            :param layers: Number of hidden layers.
        """
        super().__init__(input_dim, output_dim_encoder)
        self.hidden_layer_size = hidden_layer_size
        self.layer_norm = layer_norm
        self.lstm = nn.LSTM(input_dim, hidden_layer_size, layers)
        self.linear = nn.Linear(hidden_layer_size, output_dim_encoder)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

    def forward(self, seq_state: torch.tensor) -> torch.Tensor:
        """
        Initializes cell parameters and trains the model on given batch.
        :param omni_sequence: Three dimensional tensor of shape [seq_len, batch_size, input_size]
        :return: The predicted values of 7 channels.
        """
        # PyTorch LSTM expects input shape [seq_len, batch_size, input_size]
        lstm_out, _ = self.lstm(seq_state.permute(1, 0, 2))
        generated_features = self.linear(lstm_out)
        return generated_features[-1]

    @property
    def is_cuda(self):
        """ Returns if the model using cuda
        """
        return next(self.parameters()).is_cuda


class MLP(nn.Module):
    """
        This is a generic class for MLP.
        The architecture is customizable.
    """
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: list, activation: str, batch_norm: bool,
                 dropout: float = 0.0):

        """
        :param input_dim: number of input
        :param output_dim: number of output channel
        :param hidden_dim: list of units in hidden layer
        :param activation: type of activation function to use
        :param batch_norm: Boolean
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        current_dim = input_dim
        layers = nn.ModuleList()
        for hdim in hidden_dim:
            if batch_norm is True:
                layers.append(nn.Linear(current_dim, hdim, bias=False))
                layers.append(nn.BatchNorm1d(hdim))
            else:
                layers.append(nn.Linear(current_dim, hdim, bias=True))
            current_dim = hdim
            layers.append(activation_factory(self.activation))
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(current_dim, output_dim))
        self._model = nn.Sequential(*layers)

    def forward(self, input_x):  # pylint: disable=arguments-differ
        return self._model.forward(input_x)


class LinearEncoder(nn.Module):
    """
        This class takes an encoder and a linear model
        and stacks them together.
    """
    def __init__(self, encoder: SequenceEncoder, decoder: nn.Module):
        """
            The encoder and the linear layer is passed
            into the model.
            :param encoder: Takes in sequence of input
            and generates configurable number of hidden or
            intermediate features.
            :param decoder: Takes in hidden features and
            positional arguments as input and predicts the proton
            intensities.
        """
        super(LinearEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: tuple) -> torch.Tensor:
        """
            Calls the forward function of encoder and concatenates
            the output with positional variables which are then feed into
            the forward method of linear model.
            :param omni_seq: shape: (batch_size, seq_length, input_dim)
            :param pos: shape: (batch_size, 3)
            :return output: shape (batch_size, 7).
        """
        seq_state = x[0]
        static_state = x[1]

        # Get hidden features from encoder.
        encoder_output = self.encoder(seq_state)

        # Concatenate the hidden feature with static features.
        hidden_features = torch.cat((encoder_output, static_state), dim=1)

        # Get predictions from the decoder layer.
        output = self.decoder(hidden_features)

        return output

    @property
    def is_cuda(self):
        """ Returns if the model using cuda
        """
        return next(self.parameters()).is_cuda