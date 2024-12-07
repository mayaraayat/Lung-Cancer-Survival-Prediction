# Lung-Cancer-Survival-Prediction.

Challenge provided by OWKIN company and organized by Entrepreneur First. Bio x Hack 2024 Paris.

## Project Overview

### Objective

Aim of the challenge is to predict the survival time of a patient (predict the remaining time to live, in days) based on 3D Lung CT scans and extracted quantitative imaging features

### Resources

All images and extracted metadata features come the NSCLC-Radiomics dataset.
The tabular data are censored with the date of the end of the study (2021)

### Sample size

420 patients are included in total. 80% is allocated to the train set and 20% to the test set.

### Models

1. Daft 'Funk' Model.
This model is based on DAFT model and adpated to predict survival time for patient affected by lung cancer.
![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mayaraayat/Lung-Cancer-Survival-Prediction/blob/main/DAFT_FUNK.ipynb)

2. Death 3D CNN
This model use a 3D CNN backbone to embed CT scan and then concatenate tabular vector with the image embedding.

### Metrics
We use c-index as a metric and RandomForestForSurvivalTime as a baseline.
