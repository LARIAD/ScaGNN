import numpy as np
import torch
from torch import tensor
from typing import Union


def get_prediction_metrics(pred: tensor, gt: tensor) -> tensor:
    """
    Returns a dictionary of metrics

    Parameters
    ----------
    pred: torch.tensor
        Network predictions
    gt: torch.tensor
        Ground truth
    """

    errors = {}
    errors["error"] = torch.abs(pred - gt)

    if pred.shape[-1] == 2:
        pred_real = pred[:, 0]
        pred_imag = pred[:, 1]
        pred_ampl = torch.sqrt(pred_real**2 + pred_imag**2)
        pred_angle = torch.arctan2(pred_imag, pred_real)

        gt_real = gt[:, 0]
        gt_imag = gt[:, 1]
        gt_ampl = torch.sqrt(gt_real**2 + gt_imag**2)
        gt_angle = torch.arctan2(gt_imag, gt_real)

        errors["error_ampl"] = torch.abs(pred_ampl - gt_ampl)
        errors["error_rel_ampl"] = torch.abs(pred_ampl - gt_ampl) / gt_ampl
        errors["error_angle"] = torch.abs(torch.atan2(torch.sin(pred_angle - gt_angle), torch.cos(pred_angle - gt_angle)))

    elif pred.shape[-1] == 1:
        errors["error_rel"] = torch.abs(pred - gt) / torch.abs(gt).mean(0, keepdims=True)
    
    else:
        raise ValueError(f"pred.shape[-1] must be one of 1 or 2, not f{pred.shape[-1]}")

    return {key: torch.mean(value) for key, value in errors.items()}


def get_uncertainty_metrics(pred: tensor, uncert: tensor, gt: tensor) -> tensor:
    metrics = {}
    error = torch.abs(pred - gt)
    metrics['uncert_error'] = torch.abs(error - uncert)
    metrics['uncert_corr'] = torch.corrcoef(torch.stack([error.reshape(-1), uncert.reshape(-1)]))[0, 1]

    if uncert.shape[-1] == 2:
        uncert_real = uncert[:, 0]
        uncert_imag = uncert[:, 1]
        uncert_ampl = torch.sqrt(uncert_real**2 + uncert_imag**2)
        uncert_angle = torch.arctan2(uncert_imag, uncert_real)

        error_real = error[:, 0]
        error_imag = error[:, 1]
        error_ampl = torch.sqrt(error_real**2 + error_imag**2)
        error_angle = torch.arctan2(error_imag, error_real)

        metrics["uncert_ampl"] = torch.abs(uncert_ampl - error_ampl)
        metrics["uncert_rel_ampl"] = torch.abs(uncert_ampl - error_ampl) / error_ampl
        metrics["uncert_angle"] = torch.abs(torch.atan2(torch.sin(uncert_angle - error_angle), torch.cos(uncert_angle - error_angle)))

    elif uncert.shape[-1] == 1:
        metrics["uncert_rel"] = torch.abs(uncert - error) / torch.abs(error).mean(0, keepdims=True)
    
    else:
        raise ValueError(f"uncert.shape[-1] must be one of 1 or 2, not f{uncert.shape[-1]}")

    return {key: torch.mean(value) for key, value in metrics.items()}
