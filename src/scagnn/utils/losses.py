import torch
from torch import nn


class ErrorHuberLoss(nn.Module):
    def __init__(self, alpha=1.0, delta=1.0, reduction='mean'):
        
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.reduction = reduction

    def forward(self, preds, target):

        # 0. Separate mu and error
        mu_pred, error_pred = torch.chunk(preds, 2, dim=-1)
        
        # 1 Compute huber loss on mu predictions
        loss = torch.nn.functional.huber_loss(mu_pred, target, reduction=self.reduction)

        # 2 Compute true error
        true_error = torch.abs(mu_pred.detach() - target)

        # 3 Compute the huber_loss between the predicted and the true error
        loss += self.alpha * torch.nn.functional.huber_loss(error_pred, true_error, reduction=self.reduction)

        return loss