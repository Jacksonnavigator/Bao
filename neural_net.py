import torch
import torch.nn as nn
import torch.nn.functional as F

class BaoNet(nn.Module):
    def __init__(self, rows=4, cols=8, channels=64):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.in_ch = 2
        self.conv1 = nn.Conv2d(self.in_ch, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        # a few residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels)
            ) for _ in range(4)
        ])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * rows * cols, rows * cols)

        # value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, 2, rows, cols)
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            residual = x
            out = block(x)
            x = F.relu(out + residual)

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return p, v


def board_to_tensor(board, current_player):
    import numpy as np
    rows = len(board)
    cols = len(board[0])
    arr = np.array(board, dtype=np.float32)
    # Normalize counts (optional): divide by a small constant like 12
    norm = arr / 12.0
    # channel 0: board counts normalized
    ch0 = norm
    # channel 1: mask for current player's pits (1 if pit belongs to current player)
    ch1 = np.zeros_like(arr, dtype=np.float32)
    if current_player == 1:
        ch1[0:2, :] = 1.0
    else:
        ch1[2:4, :] = 1.0

    tensor = np.stack([ch0, ch1], axis=0)
    return torch.from_numpy(tensor)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
