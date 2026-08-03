import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import dgl
from dgl import DGLGraph
from typing import Callable, List, Tuple, Union, Dict, Any

from physicsnemo.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP, MeshGraphEdgeMLPConcat, MeshGraphEdgeMLPSum
from physicsnemo.models.gnn_layers.utils import CuGraphCSC, aggregate_and_concat
from physicsnemo.utils.profiling import profile


class RescalingInverse:
    """
    Inverse and rescale

    Parameters
    ----------
    min_x: float
        Minimum
    max_x: float
        Maximum
    """
    def __init__(self, min_x, max_x):
        self.max_inv = 1 / min_x
        self.min_inv = 1 / max_x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (1 / x - self.min_inv) / (self.max_inv - self.min_inv) * 2 -1


class PositionalEncoding:
    """
    Unidimensional sinusoidale position encoding

    Parameters
    ----------
    num_channels: int
        Number of channels
    min_dist: float
        Minimum distance
    max_dist: float
        Maximum distance
    device: str
        Device, by default 'cuda'
    """
    def __init__(self, num_channels, min_dist, max_dist, device='cuda'):
        omega_min = torch.pi / (2 * max_dist)
        omega_max = torch.pi / (2 * min_dist)
        self.div_term = (torch.exp(torch.linspace(0, 1, num_channels // 2) * np.log(omega_min / omega_max)) * omega_max).unsqueeze(0).to(device)

    def __call__(self, dist):
        return torch.cat([torch.sin(dist * self.div_term), -torch.cos(dist * self.div_term)], dim=-1)


class EdgeFeatureUpdate(nn.Module):
    """
    Update ScaGNN message-passing edge features.

    Parameters
    ----------
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        _description_, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        etype: str = None,
        level: int = None,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__()

        self.etype = etype
        self.level = level

        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat

        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_nodes,
            dst_dim=input_dim_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    @profile
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
        edge_index: int = None,
    ) -> Tensor:
        
        if edge_index is not None:
            etype = f'{self.etype}_{edge_index}'
        else:
            etype = self.etype

        # print('edge_block', etype, efeat[etype].shape, self.level, nfeat[str(self.level)].shape)
        sub_graph = dgl.edge_type_subgraph(graph, [etype])
        efeat_new = self.edge_mlp(efeat[etype], nfeat[str(self.level)], sub_graph)
        efeat[etype] = efeat_new + efeat[etype]
        return efeat, nfeat


class NodeFeatureUpdate(nn.Module):
    """
    Update ScaGNN message-passing node features.

    Parameters
    ----------
    aggregation : str, optional
        Aggregation method (sum, mean) , by default "sum"
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the node features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
       Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        etype: str,
        level_in: int = None,
        level_out: int = None,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        aggregation: str = "sum",
        recompute_activation: bool = False,
    ):
        super().__init__()

        self.etype = etype
        self.level_in = level_in
        self.level_out = level_out
        self.aggregation = aggregation

        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_nodes + input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
        edge_index: int = None,
    ) -> Tuple[Tensor, Tensor]:
        
        if edge_index is not None:
            if isinstance(self.etype, str):
                etype = f'{self.etype}_{edge_index}'
            elif isinstance(self.etype, list):
                etype = [f'{etype}_{edge_index}' for etype in self.etype]

            if len(etype) == 1:
                etype = etype[0]

        else:
            etype = self.etype
            
        if isinstance(etype, str):
            sub_graph = dgl.edge_type_subgraph(graph, [etype])
            # update edge features
            cat_feat = aggregate_and_concat(efeat[etype], nfeat[str(self.level_in)], sub_graph, self.aggregation)
        
        elif isinstance(etype, list):
            sub_graph = dgl.edge_type_subgraph(graph, etype)
            # update edge features
            cat_feat = aggregate_and_concat({etype: efeat[etype] for etype in etype}, nfeat[str(self.level_in)], sub_graph, self.aggregation)

        # update node features + residual connection
        if self.training:
            nfeat[str(self.level_out)] = nfeat[str(self.level_out)] + self.node_mlp(cat_feat) #* mask.unsqueeze(-1)

        else:
            mask = torch.logical_or(graph.ndata[f'lvl_{self.level_in}'] == 1., graph.ndata[f'lvl_{self.level_out}'] == 1.)
            nfeat[str(self.level_out)][mask] += self.node_mlp(cat_feat[mask])
        
        return efeat, nfeat


class NodeFeatureExpander(nn.Module):
    """
    Expande node features in ScaGNN downsampling blocks.

    Parameters
    ----------
    level: int
        Level before downsampling
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the node features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
       Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        level: int = None,
        input_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 0,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()

        self.level = level
        self.output_dim = output_dim

        if input_dim == output_dim:
            self.node_mlp = torch.clone

        else:
            self.node_mlp = MeshGraphMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                recompute_activation=recompute_activation,
            )

    @torch.jit.ignore()
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tuple[Tensor, Tensor]:
        
        if self.training:
            proj_nfeat = self.node_mlp(nfeat[str(self.level)])# * mask.unsqueeze(-1)
        else:
            mask = torch.logical_or(graph.ndata[f'lvl_{self.level}'] == 1., graph.ndata[f'lvl_{self.level+1}'] == 1.)
            shape = (nfeat[str(self.level)].shape[0], self.output_dim)
            device = nfeat[str(self.level)].device
            dtype = nfeat[str(self.level)].dtype
            proj_nfeat = torch.zeros(shape, device=device, dtype=dtype)
            proj_nfeat[mask] = self.node_mlp(nfeat[str(self.level)][mask])
            
        
        nfeat[str(self.level+1)] = proj_nfeat
        nfeat[f"{self.level}_proj"] = proj_nfeat.clone()

        return efeat, nfeat


class AggNodeBlock(nn.Module):
    """
    Performs U-Net style skip connection for node features in ScaGNN upsampling block

    Parameters
    ----------
    level: int
        Level before upsampling

    """

    def __init__(
        self,
        level: int = None,
    ):
        super().__init__()

        self.level = level

    @torch.jit.ignore()
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tuple[Tensor, Tensor]:
    
        mask = (graph.ndata[f"lvl_{self.level}"] == 1).unsqueeze(-1)
        nfeat[str(self.level)] = nfeat[str(self.level)] * mask + nfeat[f"{self.level-1}_proj"] * ~mask

        return efeat, nfeat
