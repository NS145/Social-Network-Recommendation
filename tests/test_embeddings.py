"""Unit tests for Stage 3: Node2Vec Representation Learning and Cosine Recommendations.
"""

from pathlib import Path
import pytest
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

from src.data_loader import load_karate_club_graph
from src.embeddings import (
    train_node2vec,
    get_node_embedding,
    extract_all_embeddings,
    save_embeddings,
    load_embeddings,
    save_node2vec_model,
    load_node2vec_model,
    compute_cosine_similarity,
    recommend_node2vec_cosine,
)


@pytest.fixture(scope="module")
def karate_graph():
    return load_karate_club_graph()


@pytest.fixture(scope="module")
def trained_node2vec_model(karate_graph):
    """Fixture providing a lightweight trained Node2Vec model."""
    return train_node2vec(
        karate_graph,
        dimensions=16,
        walk_length=8,
        num_walks=10,
        p=1.0,
        q=1.0,
        window=4,
        seed=42,
        workers=1,
        quiet=True
    )


# ============================================================================
# Test Training & Embeddings Extraction
# ============================================================================

def test_train_node2vec_output_shape(trained_node2vec_model, karate_graph):
    """Verify that model trains and produces vectors of expected dimensionality."""
    assert isinstance(trained_node2vec_model, Word2Vec)
    assert trained_node2vec_model.vector_size == 16
    
    # Check that nodes are present in word vectors
    for node in karate_graph.nodes():
        vec = get_node_embedding(trained_node2vec_model, node)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (16,)
        assert not np.all(vec == 0)


def test_train_node2vec_invalid_params():
    """Verify exceptions on invalid hyperparameters."""
    g = nx.Graph()
    with pytest.raises(ValueError, match="empty graph"):
        train_node2vec(g)
        
    g.add_edge(1, 2)
    with pytest.raises(ValueError, match="strictly positive"):
        train_node2vec(g, p=-1.0)
        
    with pytest.raises(ValueError, match="positive integers"):
        train_node2vec(g, dimensions=0)


def test_extract_all_embeddings(trained_node2vec_model, karate_graph):
    """Verify extraction dictionary maintains types and node count."""
    embeddings = extract_all_embeddings(trained_node2vec_model, karate_graph)
    assert len(embeddings) == 34
    assert 0 in embeddings
    assert isinstance(embeddings[0], np.ndarray)
    assert embeddings[0].shape == (16,)


def test_fallback_for_unknown_node(trained_node2vec_model):
    """Verify that querying a missing node returns a zero vector gracefully."""
    vec = get_node_embedding(trained_node2vec_model, 9999, dimensions=16)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (16,)
    assert np.all(vec == 0)


# ============================================================================
# Test Serialization & Deserialization
# ============================================================================

def test_embeddings_serialization_roundtrip(trained_node2vec_model, karate_graph, tmp_path):
    """Verify saving and loading embeddings dictionary."""
    embeddings = extract_all_embeddings(trained_node2vec_model, karate_graph)
    pkl_file = tmp_path / "embeddings.pkl"
    
    save_embeddings(embeddings, pkl_file)
    assert pkl_file.is_file()
    
    loaded = load_embeddings(pkl_file)
    assert len(loaded) == len(embeddings)
    np.testing.assert_array_almost_equal(loaded[0], embeddings[0])


def test_model_serialization_roundtrip(trained_node2vec_model, tmp_path):
    """Verify saving and loading complete Word2Vec model."""
    model_file = tmp_path / "model.model"
    save_node2vec_model(trained_node2vec_model, model_file)
    assert model_file.is_file()
    
    loaded_model = load_node2vec_model(model_file)
    assert loaded_model.vector_size == 16
    np.testing.assert_array_almost_equal(
        loaded_model.wv["0"],
        trained_node2vec_model.wv["0"]
    )


# ============================================================================
# Test Cosine Similarity Metric
# ============================================================================

def test_compute_cosine_similarity():
    """Verify mathematical properties of cosine similarity."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v4 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    v_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # Identical
    assert abs(compute_cosine_similarity(v1, v2) - 1.0) < 1e-6
    # Orthogonal
    assert abs(compute_cosine_similarity(v1, v3) - 0.0) < 1e-6
    # Opposite
    assert abs(compute_cosine_similarity(v1, v4) - (-1.0)) < 1e-6
    # Zero vector
    assert compute_cosine_similarity(v1, v_zero) == 0.0


# ============================================================================
# Test Recommendation Pipeline
# ============================================================================

def test_recommend_node2vec_cosine_exclusions(trained_node2vec_model, karate_graph):
    """Verify that Node2Vec recommendations strictly exclude self and existing friends."""
    user = 0
    friends = set(karate_graph.neighbors(user))
    recs = recommend_node2vec_cosine(karate_graph, trained_node2vec_model, user, top_k=5)
    
    assert len(recs) == 5
    for r in recs:
        candidate = r["candidate_id"]
        assert candidate != user, "Self-recommendation detected!"
        assert candidate not in friends, f"Existing friend {candidate} recommended!"
        assert -1.0 <= r["score"] <= 1.0
        assert "num_mutual_friends" in r
        assert "candidate_degree" in r
        
    # Check monotonic ranking
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_node2vec_isolated_node_handling():
    """Verify handling when an isolated node is queried."""
    g = nx.Graph()
    g.add_edge(1, 2)
    g.add_node(99)
    
    # Construct synthetic embeddings
    mock_embeddings = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.8, 0.2]),
        99: np.array([0.0, 0.0])  # isolated node with 0-vector
    }
    
    recs = recommend_node2vec_cosine(g, mock_embeddings, user_id=99, top_k=2)
    assert len(recs) == 2
    assert all(r["score"] == 0.0 for r in recs)
