"""Graph Neural Network (GNN) module for Link Prediction.
Implements Graph Convolutional Networks (GCN) and GraphSAGE message-passing
architectures using PyTorch and PyTorch Geometric.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score

from src.recommendation import get_candidate_users, get_mutual_friends


def prepare_pyg_link_data(
    graph_train: nx.Graph,
    feature_type: str = "identity"
) -> Tuple[Data, Dict[Any, int], Dict[int, Any]]:
    """Convert an undirected NetworkX training graph into PyTorch Geometric Data.
    
    Args:
        graph_train: Undirected NetworkX training graph (contains only training edges).
        feature_type: Node feature initialization strategy:
            - 'identity': One-hot identity matrix I_|V| (transductive representation).
            - 'degree': Normalized degree feature.
            
    Returns:
        Tuple containing:
        - data (torch_geometric.data.Data): PyG Data object with x and edge_index.
        - node_to_idx (dict): Node ID -> tensor index.
        - idx_to_node (dict): Tensor index -> Node ID.
    """
    nodes = sorted(list(graph_train.nodes()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # Message-passing edge_index (bidirectional for undirected message passing)
    src_list, dst_list = [], []
    for u, v in graph_train.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        src_list.extend([u_idx, v_idx])
        dst_list.extend([v_idx, u_idx])
        
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    
    # Node feature matrix
    if feature_type == "degree":
        degrees = [graph_train.degree(n) for n in nodes]
        max_d = max(degrees) if max(degrees) > 0 else 1
        norm_deg = [[d / max_d] for d in degrees]
        x = torch.tensor(norm_deg, dtype=torch.float)
    else:
        # One-hot identity feature
        x = torch.eye(num_nodes, dtype=torch.float)
        
    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = num_nodes
    return data, node_to_idx, idx_to_node


# ============================================================================
# GNN Architectures
# ============================================================================

class GCNLinkPredictor(nn.Module):
    """2-Layer Graph Convolutional Network (GCN) with inner-product edge decoder.
    
    Mathematical formulation:
        H^(l+1) = ReLU( \tilde{D}^(-1/2) \tilde{A} \tilde{D}^(-1/2) H^(l) W^(l) )
        Score(u, v) = sigmoid( z_u · z_v )
    """
    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 16, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index)
        return z
        
    def decode(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        u = edge_pairs[:, 0]
        v = edge_pairs[:, 1]
        logits = (z[u] * z[v]).sum(dim=-1)
        return torch.sigmoid(logits)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encode(x, edge_index)


class GraphSAGELinkPredictor(nn.Module):
    """2-Layer GraphSAGE Network with neighborhood aggregation & inner-product decoder.
    
    Mathematical formulation:
        h_N(v)^(l) = AGGREGATE( { h_u^(l-1), ∀ u ∈ N(v) } )
        h_v^(l) = ReLU( W · [ h_v^(l-1) || h_N(v)^(l) ] )
    """
    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 16, dropout: float = 0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.dropout = dropout
        
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index)
        return z
        
    def decode(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        u = edge_pairs[:, 0]
        v = edge_pairs[:, 1]
        logits = (z[u] * z[v]).sum(dim=-1)
        return torch.sigmoid(logits)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encode(x, edge_index)


# ============================================================================
# Training, Evaluation & Recommendation
# ============================================================================

def train_gnn_model(
    model: nn.Module,
    data: Data,
    node_to_idx: Dict[Any, int],
    train_pos: List[Tuple[Any, Any]],
    train_neg: List[Tuple[Any, Any]],
    epochs: int = 100,
    lr: float = 0.01,
    seed: int = 42
) -> List[float]:
    """Train a GNN model on binary link prediction loss.
    
    Args:
        model: GCNLinkPredictor or GraphSAGELinkPredictor.
        data: PyG Data object (x and edge_index).
        node_to_idx: Map of node ID -> index.
        train_pos: Positive training edge tuples.
        train_neg: Negative training edge tuples.
        epochs: Training iterations.
        lr: Learning rate.
        seed: Random seed.
        
    Returns:
        List of epoch loss values.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    # Map edge tuples to index tensors
    pos_idx = [[node_to_idx[u], node_to_idx[v]] for u, v in train_pos]
    neg_idx = [[node_to_idx[u], node_to_idx[v]] for u, v in train_neg]
    
    edge_pairs = torch.tensor(pos_idx + neg_idx, dtype=torch.long)
    labels = torch.tensor([1.0] * len(pos_idx) + [0.0] * len(neg_idx), dtype=torch.float)
    
    losses = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        z = model.encode(data.x, data.edge_index)
        preds = model.decode(z, edge_pairs)
        loss = criterion(preds, labels)
        
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        
    return losses


def evaluate_gnn_model(
    model: nn.Module,
    data: Data,
    node_to_idx: Dict[Any, int],
    test_pos: List[Tuple[Any, Any]],
    test_neg: List[Tuple[Any, Any]]
) -> Dict[str, float]:
    """Evaluate GNN link predictor on held-out test edges.
    
    Returns:
        Dict with 'roc_auc' and 'average_precision'.
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        
        pos_idx = [[node_to_idx[u], node_to_idx[v]] for u, v in test_pos]
        neg_idx = [[node_to_idx[u], node_to_idx[v]] for u, v in test_neg]
        
        edge_pairs = torch.tensor(pos_idx + neg_idx, dtype=torch.long)
        y_true = np.array([1] * len(pos_idx) + [0] * len(neg_idx))
        
        preds = model.decode(z, edge_pairs).cpu().numpy()
        
    roc_auc = float(roc_auc_score(y_true, preds))
    ap = float(average_precision_score(y_true, preds))
    return {
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(ap, 4)
    }


def recommend_gnn_links(
    graph: nx.Graph,
    model: nn.Module,
    data: Data,
    node_to_idx: Dict[Any, int],
    user_id: Any,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Generate friend recommendations using trained GNN node representations.
    
    Strictly excludes self and existing connections.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    candidates = get_candidate_users(graph, user_id)
    if not candidates:
        return []
        
    cand_list = list(candidates)
    u_idx = node_to_idx[user_id]
    cand_indices = [node_to_idx[c] for c in cand_list]
    
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        edge_pairs = torch.tensor([[u_idx, c_idx] for c_idx in cand_indices], dtype=torch.long)
        probs = model.decode(z, edge_pairs).cpu().numpy()
        
    scored_candidates = []
    for cand, prob in zip(cand_list, probs):
        mutuals = get_mutual_friends(graph, user_id, cand)
        scored_candidates.append({
            "candidate_id": cand,
            "score": round(float(prob), 4),
            "mutual_friends": mutuals,
            "num_mutual_friends": len(mutuals),
            "candidate_degree": graph.degree(cand),
        })
        
    scored_candidates.sort(key=lambda x: (-x["score"], str(x["candidate_id"])))
    top_recs = scored_candidates[:top_k]
    for rank, item in enumerate(top_recs, start=1):
        item["rank"] = rank
        
    return top_recs

