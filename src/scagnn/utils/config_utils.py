from omegaconf import ListConfig, DictConfig, OmegaConf, open_dict
import os
from typing import Any

def set_run_name(cfg: Any) -> None:
    """
    Generate a log name based on the configuration.
    
    Args:
        cfg: Configuration object containing simulation parameters.
        
    Returns:
        str: Log name.
    """
    if cfg.run_name is None:
        run_name = f"ScaGNN_{cfg.dataset_train}"

        with open_dict(cfg):
            cfg['run_name'] = run_name


def etypes_levels(cfg: Any) -> None:
    for level in range(cfg.model.num_levels-1):
        with open_dict(cfg):
            cfg.data['etypes'].append(f"down_{level}_{level+1}")
            cfg.data['etypes'].append(f"up_{level+1}_{level}")

    for etype in ['distant']:
        if etype in cfg.data.etypes:
            with open_dict(cfg):
                cfg.data['etypes'].remove(etype)

                num_new_edges = cfg.data.num_new_edges_per_nodes[etype]
                cfg.data.num_new_edges_per_nodes[etype] = 0

                for index in range(cfg.model.bottom_processor_size // cfg.model.mp_per_distant_interaction_block):
                    new_etype = f"{etype}_{index}"
                    cfg.data['etypes'].append(new_etype)
                    cfg.data.num_new_edges_per_nodes[new_etype] = num_new_edges
