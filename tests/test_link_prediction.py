"""Unit tests for Stage 4: Supervised Link Prediction using Node2Vec & Logistic Regression.
"""

from pathlib import Path
import pytest
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_karate_club_graph
from src.embeddings import train_node2vec, extract_all_embeddings
from src.link_prediction import (
    split_edges,
    sample_negative_edges,
    compute_edge_features,
    build_link_prediction_dataset,
    train_link_prediction_model,
    save_link_prediction_model,
    load_link_prediction_model,
    recommend_supervised_link_prediction,
)


@pytest.fixture(scope="module")
def karate_graph():
    return load_karate_club_graph()


@pytest.fixture(scope="module")
def prepared_splits(karate_graph):
    """Fixture providing edge splits and negative samples."""
    g_train, train_pos, test_pos = split_edges(
        karate_graph,
        test_ratio=0.2,
        seed=42,
        ensure_connected=True
    )
    train_neg, test_neg = sample_negative_edges(
        karate_graph,
        num_train_neg=len(train_pos),
        num_test_neg=len(test_pos),
        seed=42
    )
    return g_train, train_pos, test_pos, train_neg, test_neg


@pytest.fixture(scope="module")
def train_embeddings(prepared_splits):
    """Fixture training Node2Vec on g_train ONLY."""
    g_train = prepared_splits[0]
    n2v = train_node2vec(
        g_train,
        dimensions=16,
        walk_length=8,
        num_walks=15,
        seed=42,
        workers=1,
        quiet=True
    )
    return extract_all_embeddings(n2v, g_train)


# ============================================================================
# Test Data Leakage Prevention & Edge Splitting
# ============================================================================

def test_split_edges_leak_free(karate_graph):
    """Verify test edges are strictly absent from g_train."""
    g_train, train_pos, test_pos = split_edges(karate_graph, test_ratio=0.2, seed=42)
    
    assert len(train_pos) + len(test_pos) == karate_graph.number_of_edges()
    
    # Assert disjoint
    train_set = {tuple(sorted(e)) for e in train_pos}
    test_set = {tuple(sorted(e)) for e in test_pos}
    assert train_set.isdisjoint(test_set)
    
    # Assert test edges are NOT in g_train
    for u, v in test_pos:
        assert not g_train.has_edge(u, v)
        
    assert nx.is_connected(g_train)


def test_split_edges_invalid_ratio(karate_graph):
    """Verify exception on invalid test ratio."""
    with pytest.raises(ValueError, match="strictly between"):
        split_edges(karate_graph, test_ratio=1.5)


# ============================================================================
# Test Negative Sampling
# ============================================================================

def test_negative_sampling_disjoint_and_valid(karate_graph):
    """Verify negative edges do not exist in graph and train/test negatives are disjoint."""
    train_neg, test_neg = sample_negative_edges(karate_graph, num_train_neg=30, num_test_neg=10, seed=42)
    
    assert len(train_neg) == 30
    assert len(test_neg) == 10
    
    train_neg_set = {tuple(sorted(e)) for e in train_neg}
    test_neg_set = {tuple(sorted(e)) for e in test_neg}
    assert train_neg_set.isdisjoint(test_neg_set)
    
    # Assert none of the negative edges exist in the real graph
    for u, v in train_neg + test_neg:
        assert not karate_graph.has_edge(u, v)
        assert u != v  # No self-loops


# ============================================================================
# Test Feature Engineering
# ============================================================================

def test_compute_edge_features_symmetry(train_embeddings):
    """Verify symmetric edge feature operators for undirected graphs."""
    u, v = 0, 1
    for op in ["hadamard", "average", "abs_diff"]:
        feat_uv = compute_edge_features(train_embeddings, u, v, operator=op)
        feat_vu = compute_edge_features(train_embeddings, v, u, operator=op)
        np.testing.assert_array_almost_equal(feat_uv, feat_vu)
        assert feat_uv.shape == (16,)
        
    feat_concat = compute_edge_features(train_embeddings, u, v, operator="concat")
    assert feat_concat.shape == (32,)


def test_compute_edge_features_invalid_op(train_embeddings):
    with pytest.raises(ValueError, match="Unknown edge operator"):
        compute_edge_features(train_embeddings, 0, 1, operator="invalid_op")


# ============================================================================
# Test Model Training & Serialization
# ============================================================================

def test_train_and_serialize_model(prepared_splits, train_embeddings, tmp_path):
    """Verify LogisticRegression training, evaluation, and roundtrip serialization."""
    _, train_pos, test_pos, train_neg, test_neg = prepared_splits
    
    X_train, y_train = build_link_prediction_dataset(train_pos, train_neg, train_embeddings, operator="hadamard")
    X_test, y_test = build_link_prediction_dataset(test_pos, test_neg, train_embeddings, operator="hadamard")
    
    clf = train_link_prediction_model(X_train, y_train, seed=42)
    assert isinstance(clf, LogisticRegression)
    
    # Check predictions
    probs = clf.predict_proba(X_test)[:, 1]
    assert len(probs) == len(X_test)
    assert all(0.0 <= p <= 1.0 for p in probs)
    
    # Test serialization
    model_file = tmp_path / "lr_model.pkl"
    save_link_prediction_model(clf, {"operator": "hadamard"}, model_file)
    assert model_file.is_file()
    
    loaded_clf, meta = load_link_prediction_model(model_file)
    assert meta["operator"] == "hadamard"
    np.testing.assert_array_almost_equal(
        loaded_clf.predict_proba(X_test),
        clf.predict_proba(X_test)
    )


# ============================================================================
# Test Supervised Recommendation Pipeline
# ============================================================================

def test_recommend_supervised_exclusions(karate_graph, prepared_splits, train_embeddings):
    """Verify supervised recommendations strictly exclude self and existing connections."""
    _, train_pos, _, train_neg, _ = prepared_splits
    X_train, y_train = build_link_prediction_dataset(train_pos, train_neg, train_embeddings)
    clf = train_link_prediction_model(X_train, y_train, seed=42)
    
    user = 0
    friends = set(karate_graph.neighbors(user))
    recs = recommend_supervised_link_prediction(karate_graph, clf, train_embeddings, user_id=user, top_k=5)
    
    assert len(recs) == 5
    for r in recs:
        candidate = r["candidate_id"]
        assert candidate != user
        assert candidate not in friends
        assert 0.0 <= r["score"] <= 1.0
        
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)

