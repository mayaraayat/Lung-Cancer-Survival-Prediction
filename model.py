import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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


class LungCancerDataset(Dataset):
    def __init__(self, train_indices, test_indices, scans_path, demographic_path, return_train):
        self.clinical_vars = pd.load_csv(demographic_path)
        self.scans = np.load(scans_path)

        self.train_indices = train_indices
        self.test_indices = test_indices
        self.train_clinical, self.test_clinical = self.preprocess_clinical_vars(self.clinical_vars)

        if return_train:
            self.scans = torch.Tensor(self.scans[self.train_indices]).to(torch.float32).unsqueeze(0) # do we need unsqueeze?
            self.clinical_vars = torch.Tensor(self.train_clinical).to(torch.float32)
            self.events = torch.Tensor(self.clinical_vars["deadstatus.event"].values[self.train_indices]).to(torch.bool)
            self.times = torch.Tensor(self.clinical_vars["Survival.time"].values[self.train_indices]).to(torch.float32)
        else:
            self.scans = torch.Tensor(self.scans[self.test_indices]).to(torch.float32)
            self.clinical_vars = torch.Tensor(self.test_clinical).to(torch.float32)
            self.events = torch.Tensor(self.clinical_vars["deadstatus.event"].values[self.test_indices]).to(torch.bool)
            self.times = torch.Tensor(self.clinical_vars["Survival.time"].values[self.test_indices]).to(torch.float32)

    def __len__(self):
        return len(self.scans)
    
    def preprocess_clinical_vars(self, clinical_vars):

        numeric_features = ["age"]
        categorical_features = ["Histology", "gender"]
        ordinal_features = ["clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage", "Overall.Stage"]

        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        ordinal_transformer = Pipeline(
            steps=[
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
                ("ord", ordinal_transformer, ordinal_features),
            ]
        )
        train_clinical = preprocessor.fit_transform(clinical_vars)
        test_clinical = preprocessor.transform(clinical_vars)
        return train_clinical, test_clinical

    def __getitem__(self, idx):
        return (
            self.scans[idx],  # Add channel dimension
            self.events[idx],
            self.times[idx],
            self.clinical_vars[idx],
        )
    


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


def train_model(model, train_dataset, batch_size, criterion, optimizer, writer, device, num_epochs, gamma):

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader: # clinical vars too
            scans, events, times = batch
            scans, events, times = scans.to(device), events.to(device), times.to(device)

            optimizer.zero_grad()
            embedding, proba_thresh = model(scans)
            all_features = np.concatenate((embedding.detach().numpy(), clinical_vars.detach().numpy()), axis=1)
            survival_estimator = model.fit_survival_estimator(all_features, events, times)
            surv_funcs = survival_estimator.predict_survival_function(all_features)
            survival_times = compute_time_to_event(surv_funcs, thershold = proba_thresh)
            loss = criterion(survival_times, events, times, gamma)
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
