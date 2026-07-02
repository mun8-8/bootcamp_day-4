"""파킨슨병 진단 멀티모달 모델 (ECG + EMG CNN-LSTM)"""

import torch
import torch.nn as nn


class BasicConvolution(nn.Module):
    """1D Convolution (kernel=3, padding=1) + BatchNorm + LeakyReLU"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ECG_CNN(nn.Module):
    """ECG 신호용 CNN 브랜치
     입력 (B, 1, 2500) → (B, 64, 39)"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            BasicConvolution(1, 16, stride=2),
            BasicConvolution(16, 16, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            BasicConvolution(16, 32, stride=2),
            BasicConvolution(32, 32, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            BasicConvolution(32, 64, stride=2),
            BasicConvolution(64, 64, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EMG_CNN(nn.Module):
    """EMG 신호용 CNN 브랜치. 입력 (B, 1, 1250) → (B, 64, 39)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            BasicConvolution(1, 16, stride=2),
            BasicConvolution(16, 16, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            BasicConvolution(16, 32, stride=1),
            BasicConvolution(32, 32, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            BasicConvolution(32, 64, stride=2),
            BasicConvolution(64, 64, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultimodalModel(nn.Module):
    """ECG·EMG CNN 특징을 시간축(dim=2)으로 결합 후 LSTM 분류"""

    def __init__(self):
        super().__init__()
        self.ecg_cnn = ECG_CNN()
        self.emg_cnn = EMG_CNN()
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ecg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        ecg_feat = self.ecg_cnn(ecg)          # (B, 64, 39)
        emg_feat = self.emg_cnn(emg)          # (B, 64, 39)
        merged = torch.cat([ecg_feat, emg_feat], dim=2)  # (B, 64, 78)
        seq = merged.permute(0, 2, 1)         # (B, 78, 64)
        out, _ = self.lstm1(seq)
        out, (h_n, _) = self.lstm2(out)
        logits = self.fc(h_n[-1])             # (B, 1)
        return self.sigmoid(logits)


if __name__ == "__main__":
    model = MultimodalModel()
    ecg = torch.randn(4, 1, 2500)
    emg = torch.randn(4, 1, 1250)
    pred = model(ecg, emg)
    print(f"Output shape: {pred.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
