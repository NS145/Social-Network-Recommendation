"""Unit tests for Stage 8: Graph Neural Networks (GCN & GraphSAGE) Link Prediction.
Tests PyG data preparation, model architectures, training loops, evaluation, and recommendation generation.
"""

import pytest
import numpy as np
import networkx as nx
import torch

from src.data_loader import load_karate_club_graph
from src.link_prediction import split_edges, sample_negative_edges
from src.gnn import (
    prepare_pyg_link_data,
    GCNLinkPredictor,
    GraphSAGELinkPredictor,
    train_gnn_model,
    evaluate_gnn_model,
    recommend_gnn_links,
)


@pytest.fixture(scope="module")
def karate_split():
    """Shared train/test edge split for Karate Club graph."""
    g_orig = load_karate_club_graph()
    g_train, train_pos, test_pos = split_edges(g_orig, test_ratio=0.2, seed=42, ensure_connected=True)
    train_neg, test_neg = sample_negative_edges(g_orig, len(train_pos), len(test_pos), seed=42)
    return {
        "g_orig": g_orig,
        "g_train": g_train,
        "train_pos": train_pos,
        "test_pos": test_pos,
        "train_neg": train_neg,
        "test_neg": test_neg,
    }


def test_prepare_pyg_link_data_identity(karate_split):
    """Test PyG data conversion with one-hot identity features."""
    g_train = karate_split["g_train"]
    data, node_to_idx, idx_to_node = prepare_pyg_link_data(g_train, feature_type="identity")

    assert data.num_nodes == g_train.number_of_nodes()
    assert data.x.shape == (g_train.number_of_nodes(), g_train.number_of_nodes())
    # One-hot identity check
    assert torch.equal(data.x, torch.eye(g_train.number_of_nodes()))
    # Edge index should be bidirectional (2 * num_edges in g_train)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == 2 * g_train.number_of_edges()
    assert len(node_to_idx) == g_train.number_of_nodes()
    assert len(idx_to_node) == g_train.number_of_nodes()


def test_prepare_pyg_link_data_degree(karate_split):
    """Test PyG data conversion with degree features."""
    g_train = karate_split["g_train"]
    data, _, _ = prepare_pyg_link_data(g_train, feature_type="degree")

    assert data.x.shape == (g_train.number_of_nodes(), 1)
    assert torch.all(data.x >= 0.0)
    assert torch.all(data.x <= 1.0)


def test_gcn_link_predictor_forward_and_decode(karate_split):
    """Test GCN forward pass and inner-product decoding."""
    g_train = karate_split["g_train"]
    data, node_to_idx, _ = prepare_pyg_link_data(g_train, feature_type="identity")
    in_channels = data.num_nodes
    hidden_dim = 16
    out_dim = 8

    model = GCNLinkPredictor(in_channels=in_channels, hidden_channels=hidden_dim, out_channels=out_dim)
    model.eval()

    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        assert z.shape == (data.num_nodes, out_dim)

        edge_pairs = torch.tensor([[0, 1], [0, 2], [1, 3]], dtype=torch.long)
        scores = model.decode(z, edge_pairs)

        assert scores.shape == (3,)
        assert torch.all(scores >= 0.0)
        assert torch.all(scores <= 1.0)


def test_graphsage_link_predictor_forward_and_decode(karate_split):
    """Test GraphSAGE forward pass and inner-product decoding."""
    g_train = karate_split["g_train"]
    data, node_to_idx, _ = prepare_pyg_link_data(g_train, feature_type="identity")
    in_channels = data.num_nodes
    hidden_dim = 16
    out_dim = 8

    model = GraphSAGELinkPredictor(in_channels=in_channels, hidden_channels=hidden_dim, out_channels=out_dim)
    model.eval()

    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        assert z.shape == (data.num_nodes, out_dim)

        edge_pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        scores = model.decode(z, edge_pairs)

        assert scores.shape == (2,)
        assert torch.all(scores >= 0.0)
        assert torch.all(scores <= 1.0)


def test_train_gnn_model_convergence(karate_split):
    """Test GNN training loop runs and decreases loss."""
    g_train = karate_split["g_train"]
    data, node_to_idx, _ = prepare_pyg_link_data(g_train, feature_type="identity")

    model = GCNLinkPredictor(in_channels=data.num_nodes, hidden_channels=16, out_channels=8)
    epochs = 20
    losses = train_gnn_model(
        model=model,
        data=data,
        node_to_idx=node_to_idx,
        train_pos=karate_split["train_pos"],
        train_neg=karate_split["train_neg"],
        epochs=epochs,
        lr=0.02,
        seed=42
    )

    assert len(losses) == epochs
    assert all(isinstance(l, float) for l in losses)
    assert losses[-1] < losses[0], "Training loss should decrease over 20 epochs."


def test_evaluate_gnn_model_metrics(karate_split):
    """Test GNN evaluation produces valid ROC-AUC and Average Precision scores."""
    g_train = karate_split["g_train"]
    data, node_to_idx, _ = prepare_pyg_link_data(g_train, feature_type="identity")

    model = GCNLinkPredictor(in_channels=data.num_nodes, hidden_channels=16, out_channels=8)
    train_gnn_model(
        model=model,
        data=data,
        node_to_idx=node_to_idx,
        train_pos=karate_split["train_pos"],
        train_neg=karate_split["train_neg"],
        epochs=30,
        lr=0.01,
        seed=42
    )

    metrics = evaluate_gnn_model(
        model=model,
        data=data,
        node_to_idx=node_to_idx,
        test_pos=karate_split["test_pos"],
        test_neg=karate_split["test_neg"]
    )

    assert "roc_auc" in metrics
    assert "average_precision" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0
    # A trained model should easily outperform random guessing (0.50)
    assert metrics["roc_auc"] > 0.50


def test_recommend_gnn_links_structure(karate_split):
    """Test GNN recommendations exclude self, existing friends, and rank correctly."""
    g_orig = karate_split["g_orig"]
    g_train = karate_split["g_train"]
    data, node_to_idx, _ = prepare_pyg_link_data(g_train, feature_type="identity")

    model = GCNLinkPredictor(in_channels=data.num_nodes, hidden_channels=16, out_channels=8)
    train_gnn_model(
        model=model,
        data=data,
        node_to_idx=node_to_idx,
        train_pos=karate_split["train_pos"],
        train_neg=karate_split["train_neg"],
        epochs=30,
        lr=0.01,
        seed=42
    )

    user_id = 0
    top_k = 5
    recs = recommend_gnn_links(g_orig, model, data, node_to_idx, user_id=user_id, top_k=top_k)

    assert len(recs) == top_k
    existing_friends = set(g_orig.neighbors(user_id))

    scores = []
    for rank, rec in enumerate(recs, start=1):
        assert rec["rank"] == rank
        assert rec["candidate_id"] != user_id, "Candidate cannot be self."
        assert rec["candidate_id"] not in existing_friends, "Candidate cannot be an existing friend."
        assert "score" in rec
        assert "mutual_friends" in rec
        assert "num_mutual_friends" in rec
        assert "candidate_degree" in rec
        scores.append(rec["score"])

    # Monotonically descending scores
    assert scores == sorted(scores, reverse=True)


def test_recommend_gnn_links_edge_cases(karate_split):
    """Test GNN recommendation error handling and edge cases."""
    g_orig = karate_split["g_orig"]
    g_train = karate_split["g_train"]
    data, node_to_idx, _ = prepare_pyg_link_data(g_train, feature_type="identity")
    model = GCNLinkPredictor(in_channels=data.num_nodes, hidden_channels=8, out_channels=4)

    # Invalid top_k
    with pytest.raises(ValueError):
        recommend_gnn_links(g_orig, model, data, node_to_idx, user_id=0, top_k=0)

    # Clique graph where no candidates exist
    clique = nx.complete_graph(5)
    c_data, c_node_to_idx, _ = prepare_pyg_link_data(clique, feature_type="identity")
    c_model = GCNLinkPredictor(in_channels=5, hidden_channels=8, out_channels=4)
    c_recs = recommend_gnn_links(clique, c_model, c_data, c_node_to_idx, user_id=0, top_k=3)
    assert c_recs == []
