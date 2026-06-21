"""CNN model definitions used by the AG News experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def init_embedding_and_layers(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Embedding):
            nn.init.normal_(child.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                child.weight[0].fill_(0)
        elif isinstance(child, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(child.weight, nonlinearity="relu")
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class DPCNNBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual


class DPCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        num_blocks: int,
        num_classes: int = 4,
        dropout: float = 0.5,
        embedding_dropout: float = 0.2,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.region_conv = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(DPCNNBlock(num_filters) for _ in range(num_blocks))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters, num_classes)
        init_embedding_and_layers(self)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding_dropout(self.embedding(input_ids)).transpose(1, 2)
        x = self.region_conv(x)
        for block in self.blocks:
            x = block(x)
            if x.size(-1) > 2:
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: list[int],
        dropout: float,
        embedding_dropout: float,
        num_classes: int = 4,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size) for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        init_embedding_and_layers(self)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding_dropout(self.embedding(input_ids)).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            hidden = F.relu(conv(x))
            pooled.append(F.max_pool1d(hidden, kernel_size=hidden.size(-1)).squeeze(-1))
        x = self.dropout(torch.cat(pooled, dim=-1))
        return self.classifier(x)

