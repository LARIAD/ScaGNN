from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import dgl  # noqa: F401 for docs
from dgl import DGLGraph

from dataclasses import dataclass
from itertools import chain
from typing import Callable, List, Tuple, Union, Dict, Any
from warnings import warn

import physicsnemo  # noqa: F401 for docs
from physicsnemo.models.layers import get_activation
from physicsnemo.models.module import Module
from physicsnemo.utils.profiling import profile

from physicsnemo.models.meshgraphnet.meshgraphnet import MetaData, MeshGraphNetProcessor
from physicsnemo.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP, MeshGraphEdgeMLPConcat, MeshGraphEdgeMLPSum

from .edge_sampling_utils import add_new_edges_laplace, add_new_edges_helmholtz, dynamic_adaptive_edge_sampling
from .scagnn_modules import *


class ScaGNN(Module):
    """
    ScaGNN architecture.

    Parameters
    ----------
    cfg : config
        Config of the run
    input_dim_nodes : int
        Number of node features
    input_dim_edges : Dict[str, int]
        Dictionary with the number of edge features for each edge type
    output_dim : int
        Number of outputs
    num_levels : int
        Number of levels
    first_top_processor_size : int
        Number of message passing blocks in the first top processor block
    last_top_processor_size : int
        Number of message passing blocks in the last top processor block
    bottom_processor_size : int
        Number of message passing blocks in the bottom processor block
    hidden_dim_scaling: int
        Scaling factor of the hidden dimension
    mp_per_distant_interaction_block: int
        Number of message passing apply to each distant edge sample
    mlp_activation_fn : Union[str, List[str]],
        Activation function to use, by default 'relu'
    num_layers_node_processor : int, optional
        Number of MLP layers for processing nodes in each message passing block, by default 2
    num_layers_edge_processor : int, optional
        Number of MLP layers for processing edge features in each message passing block, by default 2
    hidden_dim_processor : int, optional
        Hidden layer size for the message passing blocks, by default 128
    hidden_dim_node_encoder : int, optional
        Hidden layer size for the node feature encoder, by default 128
    num_layers_node_encoder : Union[int, None], optional
        Number of MLP layers for the node feature encoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no node encoder
    hidden_dim_edge_encoder : int, optional
        Hidden layer size for the edge feature encoder, by default 128
    num_layers_edge_encoder : Union[int, None], optional
        Number of MLP layers for the edge feature encoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no edge encoder
    hidden_dim_node_decoder : int, optional
        Hidden layer size for the node feature decoder, by default 128
    num_layers_node_decoder : Union[int, None], optional
        Number of MLP layers for the node feature decoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no decoder
    aggregation: str, optional
        Message aggregation type, by default "sum"
    do_concat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    num_processor_checkpoint_segments: int, optional
        Number of processor segments for gradient checkpointing, by default 0 (checkpointing disabled)
    checkpoint_offloading: bool, optional
        Whether to offload the checkpointing to the CPU, by default False
    """

    def __init__(
        self,
        cfg: Any,
        input_dim_nodes: int,
        input_dim_edges: Dict[str, int],
        output_dim: int,
        num_levels: int,
        first_top_processor_size: int,
        last_top_processor_size: int,
        bottom_processor_size: List[int],
        hidden_dim_scaling: float,
        mp_per_distant_interaction_block: int,
        node_feature_list: list,
        edge_feature_list: list,
        mlp_activation_fn: Union[str, List[str]] = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Union[int, None] = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Union[int, None] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Union[int, None] = 2,
        aggregation: str = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        recompute_activation: bool = False,
        norm_type="LayerNorm",
    ):
        super().__init__(meta=MetaData())

        self.cfg = cfg
        self.output_dim = output_dim

        activation_fn = get_activation(mlp_activation_fn)

        if norm_type not in ["LayerNorm", "TELayerNorm"]:
            raise ValueError("Norm type should be either 'LayerNorm' or 'TELayerNorm'")

        if not torch.cuda.is_available() and norm_type == "TELayerNorm":
            warn("TELayerNorm is not supported on CPU. Switching to LayerNorm.")
            norm_type = "LayerNorm"

        max_dist = cfg.data.environment.size * np.sqrt(2) # np.sqrt(2) because of square env
        self.distance_encoding = PositionalEncoding(cfg.pe.dist.dim, cfg.data.h, max_dist)

        self.wl_encoding = RescalingInverse(cfg.data.source.wavelength.min, cfg.data.source.wavelength.max)

        self.node_feature_list = node_feature_list
        self.edge_feature_list = edge_feature_list

        self.etypes = input_dim_edges.keys()

        is_distant_edges_encoder_initialized = False

        if cfg.data.equation == 'helmholtz':
            self.add_new_edges = add_new_edges_helmholtz
        elif cfg.data.equation == 'laplace':
            self.add_new_edges = add_new_edges_laplace

        self.intermediate_decoders = cfg.intermediate_preds

        self.edge_encoders = nn.ModuleDict()
        for etype, edge_input_dim in input_dim_edges.items():
            init_edge_encoder = True
            if edge_input_dim > 0:
                if etype == 'neighbors':
                    output_dim_processor = hidden_dim_processor

                elif 'distant' in etype:
                    if not is_distant_edges_encoder_initialized:
                        is_distant_edges_encoder_initialized = True
                        output_dim_processor = int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1))
                        etype = 'distant'
                    else:
                        init_edge_encoder = False

                else:
                    for i in range(num_levels):
                        if etype in [f"down_{i}_{i+1}", f"up_{i+1}_{i}"]:
                            output_dim_processor = int(hidden_dim_processor * hidden_dim_scaling ** (i+1))
                
                # print(etype, output_dim_processor)
                if init_edge_encoder:
                    self.edge_encoders[etype] = MeshGraphMLP(
                        edge_input_dim,
                        output_dim=output_dim_processor,
                        hidden_dim=hidden_dim_edge_encoder,
                        hidden_layers=num_layers_edge_encoder,
                        activation_fn=activation_fn,
                        norm_type=norm_type,
                        recompute_activation=recompute_activation,
                    )
                
        self.node_encoder = MeshGraphMLP(
            input_dim_nodes,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        num_period = bottom_processor_size // mp_per_distant_interaction_block

        if self.intermediate_decoders:
            self.intermediate_node_decoders = nn.ModuleList()
            for _ in range(num_period):
                self.intermediate_node_decoders.append(MeshGraphMLP(
                    int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1)),
                    output_dim=output_dim,
                    hidden_dim=int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1)),
                    hidden_layers=num_layers_node_decoder,
                    activation_fn=activation_fn,
                    norm_type=None,
                    recompute_activation=recompute_activation,
                ))

        self.node_decoder = MeshGraphMLP(
            hidden_dim_processor,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )
        
        self.long_range_etypes = []
        for etype in set(input_dim_edges.keys()):
            etype = etype.split('_')[0]
            if etype in ['distant']:
                self.long_range_etypes.append(etype)
        self.long_range_etypes = list(set(self.long_range_etypes))

        self.num_levels = num_levels
        self.hidden_dim = hidden_dim_processor
        
        self.num_processor_checkpoint_segments = num_processor_checkpoint_segments
        self.checkpoint_offloading = (
            checkpoint_offloading if (num_processor_checkpoint_segments > 0) else False
        )

        edge_block_invars = dict(
            hidden_layers=num_layers_edge_processor,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=False,
        )
        node_block_invars = dict(
            hidden_layers=num_layers_node_processor,
            activation_fn=activation_fn,
            norm_type=norm_type,
            aggregation=aggregation,
            recompute_activation=False,
        )

        in_layers = []

        # Top layers
        for _ in range(first_top_processor_size):
            in_layers.append(EdgeFeatureUpdate(
                etype="neighbors",
                level=0,
                input_dim_nodes=hidden_dim_processor,
                input_dim_edges=hidden_dim_processor,
                output_dim=hidden_dim_processor,
                hidden_dim=hidden_dim_processor,
                **edge_block_invars
            ))
            in_layers.append(NodeFeatureUpdate(
                etype="neighbors",
                level_in=0,
                level_out=0,
                input_dim_nodes=hidden_dim_processor,
                input_dim_edges=hidden_dim_processor,
                output_dim=hidden_dim_processor,
                hidden_dim=hidden_dim_processor,
                **node_block_invars
            ))

        # Downward layers
        for level in range(num_levels-1):
            in_layers.append(NodeFeatureExpander(
                level=level,
                input_dim=int(hidden_dim_processor * hidden_dim_scaling ** level),
                output_dim=int(hidden_dim_processor * hidden_dim_scaling ** (level+1)),
                activation_fn=activation_fn,
                norm_type=norm_type,
                recompute_activation=False
            ))

            in_layers.append(EdgeFeatureUpdate(
                etype=f"down_{level}_{level+1}",
                level=level,
                input_dim_nodes=int(hidden_dim_processor * hidden_dim_scaling ** level),
                input_dim_edges=int(hidden_dim_processor * hidden_dim_scaling ** (level+1)),
                output_dim=int(hidden_dim_processor * hidden_dim_scaling ** (level+1)),
                hidden_dim=int(hidden_dim_processor * hidden_dim_scaling ** level),
                **edge_block_invars
            ))
            in_layers.append(NodeFeatureUpdate(
                etype=f"down_{level}_{level+1}",
                level_in=level,
                level_out=level+1,
                input_dim_nodes=int(hidden_dim_processor * hidden_dim_scaling ** level),
                input_dim_edges=int(hidden_dim_processor * hidden_dim_scaling ** (level+1)),
                output_dim=int(hidden_dim_processor * hidden_dim_scaling ** (level+1)),
                hidden_dim=int(hidden_dim_processor * hidden_dim_scaling ** (level+1)),
                **node_block_invars
            ))

        bottom_layers = []
        # Bottom layers
        for _ in range(bottom_processor_size):
            for etype in self.long_range_etypes:
                bottom_layers.append(EdgeFeatureUpdate(
                    etype=etype,
                    level=num_levels-1,
                    input_dim_nodes=int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1)),
                    input_dim_edges=int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1)),
                    output_dim=int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1)),
                    hidden_dim=int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1)),
                    **edge_block_invars
                ))
            bottom_layers.append(NodeFeatureUpdate(
                etype=self.long_range_etypes if len(self.long_range_etypes) > 1 else self.long_range_etypes[0],
                level_in=num_levels-1,
                level_out=num_levels-1,
                input_dim_nodes=int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1)),
                input_dim_edges=int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1)),
                output_dim=int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1)),
                hidden_dim=int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1)),
                **node_block_invars
            ))

        out_layers = []

        # Upward layers
        for level in range(num_levels-1, 0, -1):

            out_layers.append(AggNodeBlock(level))

            out_layers.append(EdgeFeatureUpdate(
                etype=f"up_{level}_{level-1}",
                level=level,
                input_dim_nodes=int(hidden_dim_processor * hidden_dim_scaling ** level),
                input_dim_edges=int(hidden_dim_processor * hidden_dim_scaling ** level),
                output_dim=int(hidden_dim_processor * hidden_dim_scaling ** level),
                hidden_dim=int(hidden_dim_processor * hidden_dim_scaling ** level),
                **edge_block_invars
            ))
            out_layers.append(NodeFeatureUpdate(
                etype=f"up_{level}_{level-1}",
                level_in=level,
                level_out=level-1,
                input_dim_nodes=int(hidden_dim_processor * hidden_dim_scaling ** level),
                input_dim_edges=int(hidden_dim_processor * hidden_dim_scaling ** level),
                output_dim=int(hidden_dim_processor * hidden_dim_scaling ** (level-1)),
                hidden_dim=int(hidden_dim_processor * hidden_dim_scaling ** (level-1)),
                **node_block_invars
            ))

        # Top layers
        for _ in range(last_top_processor_size):
            out_layers.append(EdgeFeatureUpdate(
                etype="neighbors",
                level=0,
                input_dim_nodes=hidden_dim_processor,
                input_dim_edges=hidden_dim_processor,
                output_dim=hidden_dim_processor,
                hidden_dim=hidden_dim_processor,
                **edge_block_invars
            ))
            out_layers.append(NodeFeatureUpdate(
                etype="neighbors",
                level_in=0,
                level_out=0,
                input_dim_nodes=hidden_dim_processor,
                input_dim_edges=hidden_dim_processor,
                output_dim=hidden_dim_processor,
                hidden_dim=hidden_dim_processor,
                **node_block_invars
            ))

        self.in_layers = nn.ModuleList(in_layers)
        self.bottom_layers = nn.ModuleList(bottom_layers)
        self.out_layers = nn.ModuleList(out_layers)

        self.mp_per_distant_interaction_block = mp_per_distant_interaction_block
        self.bottom_processor_size = bottom_processor_size
        

    def init_edge_features(self, batched_graph, etype):
        dist = batched_graph.edges[etype].data['dist']
        batched_graph.edges[etype].data['dist_pe'] = self.distance_encoding(dist)
        if 'wavelength' in batched_graph.edges[etype].data.keys():
            wavelength = batched_graph.edges[etype].data['wavelength']
            batched_graph.edges[etype].data['wavelength_pe'] = self.wl_encoding(wavelength)
        
        return torch.cat([
            batched_graph.edges[etype].data[key] for key in self.edge_feature_list
        ], dim=-1)


    @profile
    def forward(
        self,
        batched_graph,
        current_epoch=None,
        batch_num_nodes=None
    ) -> Tensor:

        if current_epoch == None:
            current_epoch = float("Inf")

        if 'source_dist' in batched_graph.ndata.keys():
            batched_graph.ndata['source_dist_pe'] = self.distance_encoding(batched_graph.ndata['source_dist'])
        if 'center_dist' in batched_graph.ndata.keys():
            batched_graph.ndata['center_dist_pe'] = self.distance_encoding(batched_graph.ndata['center_dist'])
        if 'wavelength' in batched_graph.ndata.keys():
            batched_graph.ndata['wavelength_pe'] = self.wl_encoding(batched_graph.ndata['wavelength'])
        
        node_features = torch.cat([
                batched_graph.ndata[key] for key in self.node_feature_list
            ], dim=-1)

        node_features = {'0': self.node_encoder(node_features)}

        edge_features = {}
        for etype in self.etypes:
            if etype.split('_')[0] in self.long_range_etypes:
                # edge_encoder = self.edge_encoders[etype.split('_')[0]]
                pass
            else:
                edge_encoder = self.edge_encoders[etype]
                edge_features[etype] = edge_encoder(self.init_edge_features(batched_graph, etype))
        
        for module in self.in_layers:
            edge_features, node_features = module(
                edge_features, node_features, batched_graph
            )

        pred_list = []

        if batch_num_nodes is None:
            batch_num_nodes = batched_graph.batch_num_nodes().cpu()
        
        mp_index = 0
        period_beginning = True
        for module in self.bottom_layers:
            period_index = mp_index // self.mp_per_distant_interaction_block

            if period_beginning:
                period_beginning = False

                if self.intermediate_decoders:
                    input_intermediate_decoder = node_features[f'{self.num_levels-1}']
                    if self.training:
                        preds = self.intermediate_node_decoders[period_index](input_intermediate_decoder)
                    else:
                        mask = batched_graph.ndata[f'lvl_{self.cfg.model.num_levels-1}'] == 1.
                        shape = (input_intermediate_decoder.shape[0], self.output_dim)
                        device = input_intermediate_decoder.device
                        dtype = input_intermediate_decoder.dtype
                        preds = torch.zeros(shape, device=device, dtype=dtype)
                        preds[mask] = self.intermediate_node_decoders[period_index](input_intermediate_decoder[mask])
                    
                    pred_list.append(preds)

                for etype in self.long_range_etypes:
                    edge_encoder = self.edge_encoders[etype]
                    etype = f'{etype}_{period_index}'

                    if self.cfg.data.num_new_edges_per_nodes[etype] > 0:

                        with torch.no_grad():

                            # Set the node mask
                            if self.cfg.model.num_levels > 1:
                                node_mask_level = batched_graph.ndata[f"lvl_{self.cfg.model.num_levels-1}"] == 1
                            else:
                                node_mask_level = torch.ones_like(batched_graph.ndata[f"lvl_{self.cfg.model.num_levels-1}"] == 1)

                            if not self.cfg.edge_sampling.use_error_preds or current_epoch == 0: # Ignore error predictions by intermediate decoders
                                node_count = 0
                                new_src_list = []
                                new_dst_list = []
                                for num_node in batch_num_nodes:
                                    node_mask = torch.logical_and(
                                        node_mask_level, 
                                        torch.logical_and(batched_graph.nodes() >= node_count, batched_graph.nodes() <= node_count + num_node)
                                    )
                                    new_dst, new_src = dynamic_adaptive_edge_sampling(
                                        g=batched_graph, 
                                        num_new_edges=self.cfg.data.num_new_edges_per_nodes[etype],
                                        errors=None, 
                                        candidate_edge_ratio=self.cfg.edge_sampling.candidate_edge_ratio, #2, #
                                        etype=etype, 
                                        node_mask=node_mask
                                    )
                                    new_src_list.append(new_src)
                                    new_dst_list.append(new_dst)
                                    node_count += num_node

                                self.add_new_edges(batched_graph, torch.cat(new_src_list), torch.cat(new_dst_list), etype)

                            else:
                                _, log_sigma = torch.chunk(preds, 2, dim=-1)
                                sigma = torch.exp(log_sigma).sum(-1)
                                if self.cfg.predict_normalized:
                                    sigma /= batched_graph.ndata['source_dist'].reshape(-1)
                                
                                node_count = 0
                                new_src_list = []
                                new_dst_list = []
                                for num_node in batch_num_nodes:
                                    node_mask = torch.logical_and(
                                        node_mask_level, 
                                        torch.logical_and(batched_graph.nodes() >= node_count, batched_graph.nodes() < node_count + num_node)
                                    )
                                    new_dst, new_src = dynamic_adaptive_edge_sampling(
                                        g=batched_graph, 
                                        num_new_edges=self.cfg.data.num_new_edges_per_nodes[etype],
                                        errors=sigma, 
                                        candidate_edge_ratio=self.cfg.edge_sampling.candidate_edge_ratio, 
                                        etype=etype, 
                                        node_mask=node_mask,
                                        alpha=self.cfg.edge_sampling.error_weight
                                    )

                                    new_src_list.append(new_src)
                                    new_dst_list.append(new_dst)
                                    node_count += num_node

                                self.add_new_edges(batched_graph, torch.cat(new_src_list), torch.cat(new_dst_list), etype)

                    edge_features[etype] = edge_encoder(self.init_edge_features(batched_graph, etype))

            edge_features, node_features = module(
                edge_features, node_features, batched_graph, edge_index=period_index
            )

            # Update message-passing and period counters at the end of each message-passing layer
            if isinstance(module, NodeFeatureUpdate):
                mp_index += 1
                period_ending = mp_index % self.mp_per_distant_interaction_block == 0
                if period_ending:
                    period_beginning = True

        for out_module in self.out_layers:
            edge_features, node_features = out_module(
                edge_features, node_features, batched_graph
            )
        preds = self.node_decoder(node_features['0'])
        pred_list.append(preds)

        return pred_list