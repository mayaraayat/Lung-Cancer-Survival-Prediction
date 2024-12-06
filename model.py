import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LungCancerDataset(Dataset):
    def __init__(self, scans, events, times):
        self.scans = scans  # 3D scans (NumPy array of shape [N, D, H, W])
        self.events = events  # Binary event indicators (1=event, 0=censored)
        self.times = times  # Survival times

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.scans[idx], dtype=torch.float32).unsqueeze(
                0
            ),  # Add channel dimension
            torch.tensor(self.events[idx], dtype=torch.float32),
            torch.tensor(self.times[idx], dtype=torch.float32),
        )


class TimeToDeath3DCNN(nn.Module):
    def __init__(self):
        super(TimeToDeath3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # Global pooling

        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)  # Output: Predicted survival time

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Predicted time-to-event
        return x


# Weighted Loss Function for Time-to-Death with Censoring
class CensoredMSELoss(nn.Module):
    def forward(self, predictions, events, times):
        """
        predictions: Predicted survival times (N,)
        events: Binary event indicators (N,)
        times: True survival times (N,)
        """
        uncensored_loss = torch.mean(
            (predictions[events == 1] - times[events == 1]) ** 2
        )
        censored_loss = torch.mean(
            torch.relu(predictions[events == 0] - times[events == 0]) ** 2
        )
        return uncensored_loss + 0.1 * censored_loss


def train_model(model, dataloader, criterion, optimizer, writer, device, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            scans, events, times = batch
            scans, events, times = scans.to(device), events.to(device), times.to(device)

            optimizer.zero_grad()
            predictions = model(scans).squeeze()
            loss = criterion(predictions, events, times)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_loss, epoch + 1)

    return model


def validate_model(model, dataloader, criterion, writer, device, epoch):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in dataloader:
            scans, events, times = batch
            scans, events, times = scans.to(device), events.to(device), times.to(device)

            predictions = model(scans).squeeze()
            loss = criterion(predictions, events, times)
            total_loss += loss.item()

        print(f"Validation Loss: {total_loss:.4f}")
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/Validation", avg_loss, epoch)
        return avg_loss
