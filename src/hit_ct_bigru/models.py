from __future__ import annotations

import torch
from torch import nn


class CTBiGRU(nn.Module):
    """1DCNN-Transformer-BiGRU model used as the paper's main architecture."""

    def __init__(
        self,
        input_channels: int = 4,
        num_classes: int = 3,
        conv_channels: tuple[int, int] = (64, 128),
        kernel_sizes: tuple[int, int] = (32, 16),
        strides: tuple[int, int] = (8, 4),
        paddings: tuple[int, int] = (12, 8),
        pooled_length: int = 128,
        embed_dim: int = 128,
        transformer_layers: int = 2,
        attention_heads: int = 8,
        ffn_dim: int = 512,
        transformer_dropout: float = 0.1,
        conv_dropout: float = 0.4,
        gru_hidden: int = 128,
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.5,
        return_stage_features: bool = False,
    ) -> None:
        super().__init__()
        c1, c2 = conv_channels
        k1, k2 = kernel_sizes
        s1, s2 = strides
        p1, p2 = paddings
        self.return_stage_features = return_stage_features

        self.preprocess = nn.Sequential(
            nn.Conv1d(input_channels, c1, kernel_size=k1, stride=s1, padding=p1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Dropout(conv_dropout),
            nn.Conv1d(c1, c2, kernel_size=k2, stride=s2, padding=p2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Dropout(conv_dropout),
            nn.AdaptiveAvgPool1d(pooled_length),
        )
        self.channel_projection = nn.Identity() if c2 == embed_dim else nn.Linear(c2, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, pooled_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=attention_heads,
            dim_feedforward=ffn_dim,
            dropout=transformer_dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor):
        cnn_features = self.preprocess(x)
        transformer_input = cnn_features.permute(0, 2, 1)
        transformer_input = self.channel_projection(transformer_input)
        transformer_input = transformer_input + self.positional_encoding[:, : transformer_input.size(1)]
        transformer_features = self.transformer(transformer_input)
        gru_features, _ = self.gru(transformer_features)
        final_features = gru_features.mean(dim=1)
        logits = self.classifier(final_features)
        if self.return_stage_features:
            return logits, {
                "cnn": cnn_features.detach(),
                "transformer": transformer_features.detach(),
                "bigru": gru_features.detach(),
                "final": final_features.detach(),
            }
        return logits

    def extract_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        old = self.return_stage_features
        self.return_stage_features = True
        try:
            _, features = self.forward(x)
        finally:
            self.return_stage_features = old
        return features


class BasicCNN(nn.Module):
    def __init__(self, input_channels: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=8, stride=4, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=8, stride=4, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class RawLSTM(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        num_classes: int = 3,
        hidden_size: int = 64,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
        )
        directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * directions, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        return self.classifier(x.mean(dim=1))


class CNNBiGRU(nn.Module):
    def __init__(self, input_channels: int = 4, num_classes: int = 3, gru_hidden: int = 128) -> None:
        super().__init__()
        self.cnn = CTBiGRU(
            input_channels=input_channels,
            num_classes=num_classes,
            gru_hidden=gru_hidden,
        ).preprocess
        self.gru = nn.GRU(128, gru_hidden, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x).permute(0, 2, 1)
        x, _ = self.gru(x)
        return self.classifier(x.mean(dim=1))


class CNNTransformer(nn.Module):
    def __init__(self, input_channels: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        base = CTBiGRU(input_channels=input_channels, num_classes=num_classes)
        self.cnn = base.preprocess
        self.positional_encoding = nn.Parameter(torch.zeros(1, 128, 128))
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x).permute(0, 2, 1)
        x = x + self.positional_encoding[:, : x.size(1)]
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))


class TransformerBiGRU(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        num_classes: int = 3,
        max_seq_len: int = 20480,
        downsample_factor: int = 32,
        embed_dim: int = 128,
        gru_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.downsample = nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)
        pooled_len = max_seq_len // downsample_factor
        self.input_projection = nn.Sequential(nn.Linear(input_channels, embed_dim), nn.ReLU(), nn.Dropout(0.4))
        self.positional_encoding = nn.Parameter(torch.zeros(1, pooled_len, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.gru = nn.GRU(embed_dim, gru_hidden, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x).permute(0, 2, 1)
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, : x.size(1)]
        x = self.transformer(x)
        x, _ = self.gru(x)
        return self.classifier(x.mean(dim=1))


MODEL_REGISTRY = {
    "ct_bigru": CTBiGRU,
    "cnn_trans_bigru": CTBiGRU,
    "cnn": BasicCNN,
    "lstm": RawLSTM,
    "bilstm": lambda **kwargs: RawLSTM(bidirectional=True, **kwargs),
    "cnn_bigru": CNNBiGRU,
    "cnn_transformer": CNNTransformer,
    "transformer_bigru": TransformerBiGRU,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)

