import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sksurv.linear_model import CoxPHSurvivalAnalysis
from losses import CensoredMSELoss
from utils import compute_time_to_event
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TimeToDeath3DCNN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels_conv1,
                 out_channels_conv2,
                 out_channels_conv3,
                 kernel_conv,
                 kernel_pool,
                 dropout):
        
        super(TimeToDeath3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels_conv1, kernel_size=kernel_conv, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels_conv1)
        self.pool1 = nn.MaxPool3d(kernel_size=kernel_pool, stride=2)

        self.conv2 = nn.Conv3d(out_channels_conv1, out_channels_conv2, kernel_size=kernel_conv, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels_conv2)
        self.pool2 = nn.MaxPool3d(kernel_size=kernel_pool, stride=2)

        self.conv3 = nn.Conv3d(out_channels_conv2, out_channels_conv3, kernel_size=kernel_conv, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels_conv3)
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # Global pooling

        self.fc1 = nn.Linear(out_channels_conv3, out_channels_conv2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_channels_conv2, 1)  # Output: Predicted probability threshold of survival

        self.survival_estimator = CoxPHSurvivalAnalysis()

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        embedding = self.dropout(x)
        proba_thresh = self.fc2(embedding)  # Predicted time-to-event
        proba_thresh = softmax(proba_thresh)
        return embedding, proba_thresh
    
    def fit_survival_estimator(self, all_features, events, times):
        target = np.array([(i, j) for i, j in zip(events.detach().numpy(), times.detach().numpy())])
        survival_estimator = self.survival_estimator.fit(all_features, target) # here target = (events, times)
        return survival_estimator


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
