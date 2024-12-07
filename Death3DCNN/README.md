# Lung-Cancer-Survival-Prediction

Solution:
- 3D CNN backbone with Cox survival regression prediction head.
- Loss: negative partial log-likelihood loss of Cox's proportional hazards model

3D CNN acts as an supervised feature extractor from CT scans. The probability threshold of the occurrance of death is learnt from the extracted features. The prediction head takes said threshold and the clinical/demographic variables as input to estimate survival time.

1. Install `hydra-core`
2. Check params in `experiment_config.yaml`
3. Run `run_model_exp.ipynb`