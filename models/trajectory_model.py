from typing import Tuple

import torch
from torch import nn


class TrajectoryModel(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        future_steps: int = 6,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, 2)

    def encode(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, cell) = self.encoder(inputs)
        return hidden, cell

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)
        hidden, cell = self.encode(inputs)

        decoder_input = torch.zeros(batch_size, 1, 2, device=inputs.device, dtype=inputs.dtype)
        predictions = []

        for _ in range(self.future_steps):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            step_prediction = self.output_layer(self.dropout_layer(decoder_output))
            predictions.append(step_prediction)
            decoder_input = step_prediction

        return torch.cat(predictions, dim=1)
