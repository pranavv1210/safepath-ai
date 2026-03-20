"""
Legacy synthetic-model implementation.

The unified nuScenes-native project uses `models.trajectory_model.TrajectoryModel`.
This file is kept only to avoid breaking older imports.
"""

from typing import Optional

import torch
from torch import nn


class LSTMEncoderDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 1,
        future_steps: int = 12,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = source.size(0)
        _, hidden = self.encoder(source)

        decoder_input = torch.zeros(batch_size, 1, 2, device=source.device)
        outputs = []

        for step in range(self.future_steps):
            decoded, hidden = self.decoder(decoder_input, hidden)
            prediction = self.output_layer(decoded)
            outputs.append(prediction)

            if self.training and target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, step : step + 1, :]
            else:
                decoder_input = prediction

        return torch.cat(outputs, dim=1)
