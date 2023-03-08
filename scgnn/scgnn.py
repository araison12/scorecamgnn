import collections
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList, Softmax
from torch_geometric.data import Batch, Data
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm.base import ExplainerAlgorithm
# from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (ExplainerConfig, MaskType,
                                            ModelConfig, ModelMode,
                                            ModelTaskLevel)
from torch_geometric.nn import MessagePassing, Sequential
from torch_geometric.utils import index_to_mask, k_hop_subgraph, subgraph

from .utils.embedding import get_message_passing_embeddings


class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class SCGNN(ExplainerAlgorithm):
    r"""
    The official implementation of ScoreCAM with GNN flavour [...]




    Args:

    """

    def __init__(
        self,
        depth: str = "last",
        interest_map_norm: bool = True,
        score_map_norm: bool = True,
        target_baseline="inference",
        **kwargs,
    ):
        super().__init__()
        self.depth = depth
        self.interest_map_norm = interest_map_norm
        self.score_map_norm = score_map_norm
        self.target_baseline = target_baseline
        self.name = "SCGNN"

    def supports(self) -> bool:
        task_level = self.model_config.task_level
        if task_level not in [ModelTaskLevel.graph]:
            logging.error(f"Task level '{task_level.value}' not supported")
            return False

        edge_mask_type = self.explainer_config.edge_mask_type
        if edge_mask_type not in [MaskType.object, None]:
            logging.error(f"Edge mask type '{edge_mask_type.value}' not " f"supported")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type not in [
            MaskType.common_attributes,
            MaskType.object,
            MaskType.attributes,
        ]:
            logging.error(f"Node mask type '{node_mask_type.value}' not " f"supported.")
            return False

        return True

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        target,
        **kwargs,
    ) -> Explanation:
        embedding = get_message_passing_embeddings(
            model=model, x=x, edge_index=edge_index
        )

        out = model(x=x, edge_index=edge_index)

        if self.target_baseline is None:
            c = target
        if self.target_baseline == "inference":
            c = out.argmax(dim=1).item()

        if self.depth == "last":
            score_map = self.get_score_map(
                model=model, x=x, edge_index=edge_index, emb=embedding[-1], c=c
            )
            extra_score_map = None
        elif self.depth == "all":
            score_map = self.get_score_map(
                model=model, x=x, edge_index=edge_index, emb=embedding[-1], c=c
            )
            extra_score_map = torch.cat(
                [
                    self.get_score_map(
                        model=model, x=x, edge_index=edge_index, emb=emb, c=c
                    )
                    for emb in embedding[:-1]
                ],
                dim=0,
            )
        else:
            raise ValueError(f"Depth={self.depth} not implemented yet")

        node_mask = score_map
        edge_mask = None
        node_feat_mask = None
        edge_feat_mask = None

        exp = Explanation(
            x=x,
            edge_index=edge_index,
            y=target,
            edge_mask=edge_mask,
            node_mask=node_mask,
            node_feat_mask=node_feat_mask,
            edge_feat_mask=edge_feat_mask,
            extra_score_map=extra_score_map,
        )
        return exp

    def get_score_map(
        self, model: torch.nn.Module, x: Tensor, edge_index: Tensor, emb: Tensor, c: int
    ) -> Tensor:
        interest_map = emb.clone()
        n_nodes, n_features = interest_map.size()
        score_map = torch.zeros(n_nodes).to(x.device)
        for k in range(n_features):
            _x = x.clone()
            feat = interest_map[:, k]
            if feat.min() == feat.max():
                continue
            mask = feat.clone()
            if self.interest_map_norm:
                mask = (mask - mask.min()).div(mask.max() - mask.min())
            mask = mask.reshape((-1, 1))
            _x = _x * mask
            _out = model(x=_x, edge_index=edge_index)
            _out = F.softmax(_out, dim=1)
            _out = _out.squeeze()
            val = float(_out[c])
            score_map = score_map + val * feat

        score_map = F.relu(score_map)
        if self.score_map_norm and score_map.min() != score_map.max():
            score_map = (score_map - score_map.min()).div(
                score_map.max() - score_map.min()
            )
        return score_map
