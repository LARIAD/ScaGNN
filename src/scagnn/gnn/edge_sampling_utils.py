import dgl
import torch
import numpy as np


def get_edge_dir_and_dist(src, dst, positions):
    """
    Compute the normalized direction of an edge and its length

    Parameters
    ----------
    src: torch.Tensor[int]
        List of edge source node indexes
    dst: torch.Tensor[int]
        List of edge destination node indexes
    positions: torch.Tensor[float]
        Node positions
    """ 
    rel_pos = positions[dst] - positions[src]
    dist = torch.linalg.norm(rel_pos, axis=-1, keepdim=True)
    rel_dir = rel_pos / dist
    return rel_dir, dist 


def remove_multi_edges(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove duplicate edges using vectorized GPU operations, no CPU-GPU sync.

    Parameters
    ----------
    src: torch.Tensor[int]
        List of edge source node indexes
    dst: torch.Tensor[int]
        List of edge destination node indexes
    """
    if len(src) > 0:
        # max_node stays as a GPU tensor — never call .item()
        max_node = torch.maximum(src.max(), dst.max()) + 1
        edge_ids = src * max_node + dst

        # Stable sort keeps first-occurrence at the front for each duplicate group
        sorted_ids, sort_idx = torch.sort(edge_ids, stable=True)

        # Mask: True wherever a new unique edge begins (fully on GPU)
        first_occurrence = torch.cat([
            torch.ones(1, dtype=torch.bool, device=src.device),
            sorted_ids[1:] != sorted_ids[:-1]
        ])

        # Map back to original positions, then re-sort to restore insertion order
        unique_indices, _ = torch.sort(sort_idx[first_occurrence])

        return src[unique_indices], dst[unique_indices]

    return src, dst


def dynamic_adaptive_edge_sampling(g: dgl.DGLGraph, 
    num_new_edges: float, 
    errors: torch.Tensor = None, 
    candidate_edge_ratio: int = 1, 
    etype: str = 'neighbors', 
    node_mask: torch.Tensor = None, 
    alpha: float = 0.
) -> None:
    """
    New edge selection using the Dynamic Adaptive Edge Sampling.

    Parameters
    ----------
    num_new_edges: float
        Number of new edges per node if >= 1.
        Otherwise, the number of edges is set to number of nodes at the lowest level * num_new_edges
    errors: torch.Tensor
        Error predictions returned by the intermediate decoders
    candidate_edge_ratio: int
        Number of candidate edge per new edges to generate
    etype: str
        the edge type
    node_mask: torch.tensor
        Node mask to select only the nodes at the lowest level
    alpha: float
        hyperparameter of the score function
    """
    
    if 'distant' in etype:
        etype = 'distant'

    orig_nodes = g.nodes()[node_mask] if node_mask is not None else g.nodes()
    num_nodes = len(orig_nodes)
    new_nodes = torch.arange(num_nodes, device=g.device)
    
    positions = g.ndata['position'][node_mask] if node_mask is not None else g.ndata['position']

    if etype in ['neighbors', 'distant']:
        num_candidate_edges = num_new_edges * num_nodes if num_new_edges < 1 else num_new_edges

        if num_candidate_edges < 1:
            new_src = torch.randperm(num_nodes, dtype=torch.int32, device=g.device)[:int(num_nodes * num_candidate_edges)]
        else:
            num_candidate_edges = int(np.round(num_candidate_edges))
            new_src = new_nodes.unsqueeze(-1).repeat(1, num_candidate_edges).reshape(-1)

        if candidate_edge_ratio > 1:
            new_dst = torch.randint(high=num_nodes, size=(num_nodes * num_candidate_edges, candidate_edge_ratio), dtype=torch.int32, device=g.device)

            if errors is not None and alpha > 0:
                errors = errors[node_mask] if node_mask is not None else errors
                if alpha < 1000:
                    new_error = errors[new_dst]
                    
                    new_rel_pos = positions[new_dst] - positions[new_src].reshape(-1, 1, 3)
                    new_dist = torch.linalg.norm(new_rel_pos, axis=-1)

                    edges_to_keep = torch.min(new_dist / (new_error ** alpha), dim=-1)[1]
                else:
                    new_error = errors[new_dst]
                    edges_to_keep = torch.max(new_error, dim=-1)[1]

            else:
                new_rel_pos = positions[new_dst] - positions[new_src].reshape(-1, 1, 3)
                new_dist = torch.linalg.norm(new_rel_pos, axis=-1)
                edges_to_keep = torch.min(new_dist, dim=-1)[1]

            new_dst = torch.gather(new_dst, dim= 1, index=edges_to_keep.unsqueeze(-1)).squeeze(-1)
            
        else:
            new_dst = torch.randint_like(new_src, low=0, high=num_nodes, dtype=torch.int32, device=g.device)

    # remove self loops
    self_loops = new_src == new_dst
    new_src, new_dst = new_src[~self_loops], new_dst[~self_loops]

    # remove multi edges
    if len(new_src) > 0:
        new_src, new_dst = remove_multi_edges(new_src, new_dst)
    
    return orig_nodes[new_src], orig_nodes[new_dst]


def add_new_edges_laplace(g: dgl.DGLGraph, new_src, new_dst, etype: str) -> dgl.DGLGraph:
    """
    Creating new edges and their input features for Laplace problems

    Parameters
    ----------
    g: dgl.DGLGraph
        The graph
    new_src: torch.Tensor[int]
        The new edge source node indexes
    new_dst: torch.Tensor[int]
        The new edge destination node indexes
    etype: str
    """
    positions = g.ndata['position']

    rel_dir, dist = get_edge_dir_and_dist(new_src, new_dst, positions)
    data_new_edges = {
        'rel_dir': rel_dir,
        'dist': dist,
        # 'amplitudes': torch.tensor(g.ndata['amplitudes'][new_src]).view(1, -1) * torch.ones((len(new_src), 1), device=dist.device),
        'amplitudes': g.ndata['amplitudes'][new_src],
    }

    g.add_edges(
        new_src, 
        new_dst, 
        data=data_new_edges, 
        etype=etype
    )
    return


def add_new_edges_helmholtz(g: dgl.DGLGraph, new_src, new_dst, etype: str) -> dgl.DGLGraph:
    """
    Creating new edges and their input features for Helmholtz problems

    Parameters
    ----------
    g: dgl.DGLGraph
        The graph
    new_src: torch.Tensor[int]
        The new edge source node indexes
    new_dst: torch.Tensor[int]
        The new edge destination node indexes
    etype: str
    """
    positions = g.ndata['position']

    rel_dir, dist = get_edge_dir_and_dist(new_src, new_dst, positions)
    data_new_edges = {
        'rel_dir': rel_dir,
        'dist': dist,
        'phase': torch.cat([
                torch.sin(2 * torch.pi / g.ndata['wavelength'][new_src] * dist), 
                torch.cos(2 * torch.pi / g.ndata['wavelength'][new_src] * dist)
        ], axis=-1),
        'wavelength': g.ndata['wavelength'][new_src] * torch.ones((len(new_src), 1), device=dist.device),
    }

    g.add_edges(
        new_src, 
        new_dst, 
        data=data_new_edges, 
        etype=etype
    )
    return