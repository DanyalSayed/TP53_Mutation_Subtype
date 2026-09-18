"""SimpleCNN: window-size-agnostic 1D CNN for TP53 mutation subtype
classification. Reused unchanged by notebooks/01_main_experiment.ipynb
(window=21) and notebooks/02_window_ablation.ipynb (window=11/51/101).

7,206 trainable parameters:
  Conv1d(4->32, k=3, same padding)  : 4*32*3 + 32   =   416
  BatchNorm1d(32)                   : 32 + 32        =    64
  Conv1d(32->64, k=3, same padding) : 32*64*3 + 64   =  6208
  BatchNorm1d(64)                   : 64 + 64        =   128
  Linear(64->6)                     : 64*6 + 6       =   390
                                                total =  7206
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """x: (batch, 4, seq_len) one-hot encoded, A=0,C=1,G=2,T=3.

        Returns raw logits of shape (batch, num_classes). Apply softmax
        separately at inference time to get class probabilities (training
        uses nn.CrossEntropyLoss, which expects logits).
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = x.amax(dim=2)  # global max pool across the sequence dimension
        x = self.dropout(x)
        return self.fc(x)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
