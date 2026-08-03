from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from multiprocessing import cpu_count
import json
import bempp_cl.api
from copy import deepcopy

from torch.optim.lr_scheduler import CosineAnnealingLR

from gnn import *
from utils.losses import ErrorHuberLoss
from utils.metrics import get_prediction_metrics, get_uncertainty_metrics

from data.data_utils import *


def normalize(feature, dist=None):
    if dist is not None:
        return feature * dist
    return feature


def denormalize(feature, dist=None):
    if dist is not None:
        return feature / dist
    return feature


class ScaGNNLightning(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        dim_obs = sum(cfg.data.node_features.values())
        dim_edges = sum(cfg.data.edge_features.values())

        try:
            if cfg.data.equation == "helmholtz":
                num_out = 2
            elif cfg.data.equation == "laplace":
                num_out = 1
            else:
                raise ValueError(f"Wrong equation name '{cfg.data.equation}'")
            
        except:
            print('Assumed equation: Helmholtz')
            num_out = 2

        if cfg.intermediate_preds:
            num_out *= 2

        # Initialize model
        self.model = ScaGNN(
            cfg=cfg,
            num_levels=cfg.model.num_levels,
            input_dim_nodes=dim_obs,
            input_dim_edges={etype: dim_edges for etype in cfg.data.etypes},
            output_dim=num_out,
            first_top_processor_size=cfg.model.first_top_processor_size,
            last_top_processor_size=cfg.model.last_top_processor_size,
            bottom_processor_size=cfg.model.bottom_processor_size,
            mp_per_distant_interaction_block=cfg.model.mp_per_distant_interaction_block,
            hidden_dim_processor=cfg.model.hidden_dim,
            hidden_dim_node_encoder=cfg.model.hidden_dim,
            hidden_dim_edge_encoder=cfg.model.hidden_dim,
            hidden_dim_node_decoder=cfg.model.hidden_dim,
            hidden_dim_scaling=cfg.model.hidden_dim_scaling,
            mlp_activation_fn='silu',
            node_feature_list=cfg.data.node_features,
            edge_feature_list=cfg.data.edge_features,
        )
        
        self.loss_preds = torch.nn.HuberLoss()
        self.loss_errors = ErrorHuberLoss()
        self.loss_factor = cfg.loss_factor
        
        self.save_hyperparameters()


    def configure_optimizers(self):
        if self.cfg.lr_scheduler:
            # optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr_max)
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr_max, weight_decay=1e-5)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.max_epochs, eta_min=self.cfg.lr_min)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch', # Call scheduler.step() every epoch
                    'frequency': 1,
                },
            }
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr_max)


    def forward_step(self, batched_graph, current_epoch=None):

        out = self.model(batched_graph, current_epoch)
        
        return out


    def training_step(self, batched_graph, batch_idx):

        preds = self.forward_step(batched_graph, current_epoch=self.current_epoch)

        labels = batched_graph.ndata['labels']

        if self.cfg.intermediate_preds:
            for i, pred in enumerate(preds):
                mu, log_sigma = torch.chunk(pred, 2, dim=-1)
                preds[i] = torch.cat([mu, torch.exp(log_sigma)], dim=-1)

            labels_intermediate = labels[batched_graph.ndata[f'lvl_{self.cfg.model.num_levels-1}'] == 1.]

        if 'source_dist' in batched_graph.ndata.keys():
            source_dist = batched_graph.ndata['source_dist']
        else:
            source_dist = None
        
        # Get normalized and denormalized preds
        if self.cfg.predict_normalized:
            preds_normalized = preds
            preds = [denormalize(pred, source_dist) for pred in preds]
        else:
            preds_normalized = [normalize(pred, source_dist) for pred in preds]

        if self.cfg.intermediate_preds:
            
            for i, (pred, pred_normalized) in enumerate(zip(preds[:-1], preds_normalized[:-1])):
                preds[i] = pred[batched_graph.ndata[f'lvl_{self.cfg.model.num_levels-1}'] == 1.]
                preds_normalized[i] = pred_normalized[batched_graph.ndata[f'lvl_{self.cfg.model.num_levels-1}'] == 1.]

            losses = []
            for pred in preds[:-1]:
                losses.append(self.loss_errors(pred, labels_intermediate))
            losses.append(self.loss_preds(torch.chunk(preds[-1], 2, dim=-1)[0], labels))
        else:
            losses = [self.loss_preds(pred, labels) for pred in preds]
        loss = sum([self.loss_factor ** i * loss for i, loss in enumerate(losses[::-1])])
        self.log("train/loss", loss, on_epoch=True, batch_size=self.cfg.training_batch_size)
        
        if 'labels_normalized' in batched_graph.ndata.keys():
            labels_normalized = batched_graph.ndata['labels_normalized']
            if self.cfg.intermediate_preds:
                losses_normalized = []
                labels_normalized_intermediate = labels_normalized[batched_graph.ndata[f'lvl_{self.cfg.model.num_levels-1}'] == 1.]
                for pred in preds_normalized[:-1]:
                    losses_normalized.append(self.loss_errors(pred, labels_normalized_intermediate))
                losses_normalized.append(self.loss_preds(torch.chunk(preds_normalized[-1], 2, dim=-1)[0], labels_normalized))
            else:
                losses_normalized = [self.loss_preds(pred, labels_normalized) for pred in preds_normalized]
            loss_normalized = sum([self.loss_factor ** i * loss for i, loss in enumerate(losses_normalized[::-1])])
            self.log("train/loss_norm", loss_normalized, on_epoch=True, batch_size=self.cfg.training_batch_size)
        
        if self.cfg.intermediate_preds and len(preds) >= 2:
            pred, sigma = torch.chunk(preds[-2], 2, dim=-1)
            uncert_metrics = get_uncertainty_metrics(
                pred, 
                sigma, 
                labels_intermediate
            )
            for metric, value in uncert_metrics.items():
                self.log(f"train/{metric}", value, on_epoch=True, batch_size=self.cfg.training_batch_size)
            pred, _ = torch.chunk(preds[-1], 2, dim=-1)
        else:
            pred = preds[-1]
        metrics = get_prediction_metrics(pred, labels)
        for metric, value in metrics.items():
            self.log(f"train/{metric}", value, on_epoch=True, batch_size=self.cfg.training_batch_size)

        if self.cfg.compare_normalized:
            return loss_normalized
        return loss


    def validation_step(self, batched_graph, batch_idx):

        preds = self.forward_step(batched_graph)
        
        labels = batched_graph.ndata['labels']

        if self.cfg.intermediate_preds:
            for i, pred in enumerate(preds):
                mu, log_sigma = torch.chunk(pred, 2, dim=-1)
                preds[i] = torch.cat([mu, torch.exp(log_sigma)], dim=-1)

            labels_intermediate = labels[batched_graph.ndata[f'lvl_{self.cfg.model.num_levels-1}'] == 1.]
        
        if 'source_dist' in batched_graph.ndata.keys():
            source_dist = batched_graph.ndata['source_dist']
        else:
            source_dist = None
        
        # Get normalized and denormalized preds
        if self.cfg.predict_normalized:
            preds_normalized = preds
            preds = [denormalize(pred, source_dist) for pred in preds]
        else:
            preds_normalized = [normalize(pred, source_dist) for pred in preds]

        if self.cfg.intermediate_preds:
            
            for i, (pred, pred_normalized) in enumerate(zip(preds[:-1], preds_normalized[:-1])):
                preds[i] = pred[batched_graph.ndata[f'lvl_{self.cfg.model.num_levels-1}'] == 1.]
                preds_normalized[i] = pred_normalized[batched_graph.ndata[f'lvl_{self.cfg.model.num_levels-1}'] == 1.]

            losses = []
            for pred in preds[:-1]:
                losses.append(self.loss_errors(pred, labels_intermediate))
            losses.append(self.loss_preds(torch.chunk(preds[-1], 2, dim=-1)[0], labels))
        else:
            losses = [self.loss_preds(pred, labels) for pred in preds]
        loss = sum([self.loss_factor ** i * loss for i, loss in enumerate(losses[::-1])])
        self.log("val/loss", loss, on_epoch=True, batch_size=self.cfg.val_batch_size)

        if 'labels_normalized' in batched_graph.ndata.keys():
            labels_normalized = batched_graph.ndata['labels_normalized']
            if self.cfg.intermediate_preds:
                losses_normalized = []
                labels_normalized_intermediate = labels_normalized[batched_graph.ndata[f'lvl_{self.cfg.model.num_levels-1}'] == 1.]
                for pred in preds_normalized[:-1]:
                    losses_normalized.append(self.loss_errors(pred, labels_normalized_intermediate))
                losses_normalized.append(self.loss_preds(torch.chunk(preds_normalized[-1], 2, dim=-1)[0], labels_normalized))
            else:
                losses_normalized = [self.loss_preds(pred, labels_normalized) for pred in preds_normalized]
            loss_normalized = sum([self.loss_factor ** i * loss for i, loss in enumerate(losses_normalized[::-1])])
            self.log("val/loss_norm", loss_normalized, on_epoch=True, batch_size=self.cfg.val_batch_size)

        if self.cfg.intermediate_preds and len(preds) >= 2:
            pred, sigma = torch.chunk(preds[-2], 2, dim=-1)
            uncert_metrics = get_uncertainty_metrics(
                pred, 
                sigma, 
                labels_intermediate
            )
            for metric, value in uncert_metrics.items():
                self.log(f"val/{metric}", value, on_epoch=True, batch_size=self.cfg.val_batch_size)
            pred, _ = torch.chunk(preds[-1], 2, dim=-1)
        else:
            pred = preds[-1]
        metrics = get_prediction_metrics(pred, labels)
        for metric, value in metrics.items():
            self.log(f"val/{metric}", value, on_epoch=True, batch_size=self.cfg.val_batch_size)


    def test_step(self, batched_data, batch_idx):

        batched_graph = batched_data

        preds = self.forward_step(batched_graph)

        labels = batched_graph.ndata['labels']

        if self.cfg.intermediate_preds:
            for i, pred in enumerate(preds):
                mu, log_sigma = torch.chunk(pred, 2, dim=-1)
                preds[i] = torch.cat([mu, torch.exp(log_sigma)], dim=-1)

            labels_intermediate = labels[batched_graph.ndata[f'lvl_{self.cfg.model.num_levels-1}'] == 1.]

        if 'source_dist' in batched_graph.ndata.keys():
            source_dist = batched_graph.ndata['source_dist']
        else:
            source_dist = None
        
        # Get normalized and denormalized preds
        if self.cfg.predict_normalized:
            preds = [denormalize(pred, source_dist) for pred in preds]

        if self.cfg.intermediate_preds:
            for i, pred in enumerate(preds[:-1]):
                preds[i] = pred[batched_graph.ndata[f'lvl_{self.cfg.model.num_levels-1}'] == 1.]

        if self.cfg.intermediate_preds and len(preds) >= 2:
            pred, sigma = torch.chunk(preds[-2], 2, dim=-1)
            uncert_metrics = get_uncertainty_metrics(
                pred, 
                sigma, 
                labels_intermediate
            )
            for metric, value in uncert_metrics.items():
                self.log(f"test/{metric}", value, on_epoch=True, batch_size=self.cfg.val_batch_size)
            pred, _ = torch.chunk(preds[-1], 2, dim=-1)
        else:
            pred = preds[-1]
        metrics = get_prediction_metrics(pred, labels)
        for metric, value in metrics.items():
            self.log(f"test/{metric}", value, on_epoch=True, batch_size=self.cfg.val_batch_size)
