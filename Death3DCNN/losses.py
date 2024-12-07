import torch
import torch.nn as nn


class CensoredMSELoss(nn.Module):
    def forward(self, predictions, events, times, gamma):
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
        return uncensored_loss + gamma * censored_loss


class CoxPHLoss(nn.Module):
    """
    Negative partial log-likelihood of Cox's proportional hazards model.
    """

    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, y_true, y_pred):
        """
        Compute loss.

        Parameters
        ----------
        y_true : tuple of Tensors
            - The first element holds a binary vector where 1 indicates an event, 0 indicates censoring.
            - The second element holds the risk set, a boolean matrix where the `i`-th row denotes the
              risk set of the `i`-th instance, i.e., indices `j` where the observed time `y_j >= y_i`.
        y_pred : Tensor
            The predicted outputs. Must be a rank 2 tensor with shape (batch_size, 1).

        Returns
        -------
        loss : Tensor
            Loss for the batch.
        """
        event, riskset = y_true  # Unpack y_true
        predictions = y_pred

        # Assertions for input shapes and types
        if predictions.ndim != 2 or predictions.size(1) != 1:
            raise ValueError(
                f"predictions must be of shape (batch_size, 1), but got {predictions.shape}."
            )

        if event.ndim != predictions.ndim:
            raise ValueError(
                f"event and predictions must have the same number of dimensions, "
                f"but got {event.ndim} and {predictions.ndim}."
            )

        if riskset.ndim != 2:
            raise ValueError(f"riskset must have 2 dimensions, but got {riskset.ndim}.")

        # Ensure data types match
        event = event.to(predictions.dtype)
        riskset = riskset.bool()  # Ensure riskset is a boolean tensor

        # Normalize predictions (optional; equivalent to `safe_normalize` in TF)
        predictions = predictions - predictions.mean(dim=0, keepdim=True)

        # Transpose predictions for matrix operations
        pred_t = predictions.T  # Shape: (1, batch_size)

        # Compute log-sum-exp over the risk set
        # Masked log-sum-exp
        riskset_masked = torch.where(
            riskset, pred_t, torch.tensor(float("-inf"), device=pred_t.device)
        )
        logsumexp = torch.logsumexp(
            riskset_masked, dim=1, keepdim=True
        )  # Shape: (batch_size, 1)

        # Compute the Cox partial log-likelihood loss
        losses = event * (logsumexp - predictions)  # Shape: (batch_size, 1)

        # Return mean loss over the batch
        return losses.mean()
