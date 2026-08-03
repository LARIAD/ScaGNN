from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.dataloading import GraphDataLoader
import hydra
from omegaconf import DictConfig
from scagnn_lightning_module import ScaGNNLightning
import wandb
from pathlib import Path

from data import ScaGNNDatasetTest
from utils.config_utils import set_run_name, etypes_levels

# Main testing script with Hydra
@hydra.main(version_base=None, config_path="../../configs/scagnn", config_name="testing")
def main(cfg: DictConfig):
    etypes_levels(cfg)
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)
    
    test_dataset = ScaGNNDatasetTest(os.path.join(cfg.dataset_root, cfg.dataset_test), cfg)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=1, drop_last=False, num_workers=6)

    checkpoint_reference = cfg.checkpoint_reference

    run = wandb.init(project="scagnn")
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()

    lightning_model = ScaGNNLightning.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

    with open_dict(lightning_model.cfg):
        lightning_model.cfg.edge_sampling.candidate_edge_ratio = cfg.edge_sampling.candidate_edge_ratio
    
    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer()

    # Test the model
    trainer.test(lightning_model, test_dataloader)

if __name__ == "__main__":
    main()