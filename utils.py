import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import torch
import os

def compute_time_to_event(surv_funcs, threshold=0.8):
    time_to_event = []
    for surv_func in surv_funcs:
        # Find the time where survival probability <= threshold
        time_idx = np.where(surv_func.y <= threshold)[0]
        if len(time_idx) > 0:
            time_to_event.append(surv_func.x[time_idx[0]])  # First time below threshold
        else:
            time_to_event.append(np.inf)  # Event hasn't occurred
    return time_to_event

class LungCancerDataset(Dataset):
    def __init__(self, scans_path_train, scans_path_test, clinical_path, return_train):
        clinical_vars = pd.read_csv(clinical_path)
        
        train_files = os.listdir(scans_path_train)
        test_files = os.listdir(scans_path_test)

        subjects_train = [i[:3] for i in train_files]
        subjects_test = [i[:3] for i in test_files]

        subjects_scans = np.unique(np.array(subjects_train + subjects_test))
        subjects_clinical  = clinical_vars["PatientID"].apply(lambda x: x[:3]).values

        for subject in subjects_scans:
            if subject not in subjects_clinical:
                clinical_vars[clinical_vars['PatientID'].str.contains(subject, na=False)]

        self.train_clinical = clinical_vars[clinical_vars['PatientID'].str.contains('|'.join(subjects_train), na=False)]
        self.test_clinical = clinical_vars[clinical_vars['PatientID'].str.contains('|'.join(subjects_test), na=False)]

        if return_train:
            self.scans = torch.Tensor([iio.imread(os.path.join(scans_path_train, file)) for file in train_files]).to(torch.float32).unsqueeze(0)

            self.events = torch.Tensor(self.train_clinical["deadstatus.event"].values).to(torch.bool)
            self.times = torch.Tensor(self.train_clinical["Survival.time"].values).to(torch.float32)
            self.train_clinical = self.train_clinical.drop(columns=["deadstatus.event", "Survival.time"])
            self.test_clinical = self.test_clinical.drop(columns=["deadstatus.event", "Survival.time"])
            self.clinical_vars = torch.Tensor(self.preprocess_clinical_vars()[0]).to(torch.float32)
        else:
            self.scans = torch.Tensor([iio.imread(os.path.join(scans_path_test, file)) for file in test_files]).to(torch.float32).unsqueeze(0)

            self.events = torch.Tensor(self.test_clinical["deadstatus.event"].values).to(torch.bool)
            self.times = torch.Tensor(self.test_clinical["Survival.time"].values).to(torch.float32)
            self.train_clinical = self.train_clinical.drop(columns=["deadstatus.event", "Survival.time"])
            self.test_clinical = self.test_clinical.drop(columns=["deadstatus.event", "Survival.time"])
            self.clinical_vars = torch.Tensor(self.preprocess_clinical_vars()[1]).to(torch.float32)
            
    def __len__(self):
        return len(self.scans)
    
    def preprocess_clinical_vars(self):

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
        train_clinical = preprocessor.fit_transform(self.train_clinical)
        test_clinical = preprocessor.transform(self.test_clinical)
        return train_clinical, test_clinical

    def __getitem__(self, idx):
        return (
            self.scans[idx],  # Add channel dimension
            self.events[idx],
            self.times[idx],
            self.clinical_vars[idx],
        )