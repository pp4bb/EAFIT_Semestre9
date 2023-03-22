# add to python path
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent.parent / "data" / "ourMNIST"))

from load import load_all

from lenet.idx import parse_idx


class IDXLeNetDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str, padding: int = 2):
        self.images = parse_idx(images_path)
        self.labels = parse_idx(labels_path)
        self.padding = padding

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = (
            F.pad(
                torch.tensor(self.images[index], dtype=torch.float32),
                pad=[self.padding] * 4,
                mode="constant",
                value=0,
            )
            / 255
        )

        # add extra dimension for channels
        img = img.unsqueeze(0)

        return img, torch.tensor(self.labels[index], dtype=torch.long)


class OurMNISTDataset(Dataset):
    def __init__(self, padding: int = 0):
        self.images, self.label = load_all()
        self.padding = padding

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = F.pad(
            torch.tensor(self.images[index], dtype=torch.float32),
            pad=[self.padding] * 4,
            mode="constant",
            value=0,
        )

        # add extra dimension for channels
        img = img.unsqueeze(0)

        return img, torch.tensor(self.label[index], dtype=torch.long)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # disable bias
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
