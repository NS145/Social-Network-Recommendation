"""Unit tests for Stage 7: Real-world Dataset Integration (Stanford SNAP Facebook).
"""

from pathlib import Path
import pytest
import networkx as nx

from src.data_loader import download_snap_facebook_dataset, load_snap_facebook_graph
from src.graph_builder import validate_graph
from src.graph_analysis import compute_network_statistics
from src.recommendation import get_candidate_users, recommend_common_neighbors


# ============================================================================
# Test Dataset Loading & Ego Subgraphs
# ============================================================================

def test_download_snap_facebook_dataset():
    """Verify SNAP Facebook dataset file exists and has non-zero size."""
    file_path = download_snap_facebook_dataset()
    assert file_path.is_file()
    assert file_path.stat().st_size > 500_000  # Should be ~850KB


def test_load_facebook_full_graph():
    """Verify loading full Facebook graph topology."""
    g = load_snap_facebook_graph()
    assert isinstance(g, nx.Graph)
    assert not g.is_directed()
    assert g.number_of_nodes() == 4039
    assert g.number_of_edges() == 88234
    
    val = validate_graph(g)
    assert val["is_valid"] is True
    assert val["num_self_loops"] == 0


def test_load_facebook_ego_graph():
    """Verify loading specific ego network (Ego 0)."""
    g_ego0 = load_snap_facebook_graph(ego_user=0)
    assert isinstance(g_ego0, nx.Graph)
    assert g_ego0.number_of_nodes() == 348
    assert g_ego0.number_of_edges() == 2866
    assert 0 in g_ego0
    
    val = validate_graph(g_ego0)
    assert val["is_valid"] is True
    assert val["is_connected"] is True


def test_invalid_ego_user():
    """Verify exception for non-existent ego user."""
    with pytest.raises(ValueError, match="does not exist"):
        load_snap_facebook_graph(ego_user=999999)


# ============================================================================
# Test Graph Statistics & Recommendations on SNAP Facebook
# ============================================================================

def test_facebook_ego0_statistics():
    """Verify structural metrics of Facebook Ego 0."""
    g_ego0 = load_snap_facebook_graph(ego_user=0)
    stats = compute_network_statistics(g_ego0)
    
    assert stats["num_nodes"] == 348
    assert stats["num_edges"] == 2866
    assert abs(stats["density"] - 0.0475) < 0.001
    assert stats["is_connected"] is True
    assert stats["average_clustering_coefficient"] > 0.5


def test_facebook_recommendation_pipeline():
    """Verify recommendation exclusions and output on Facebook Ego 0."""
    g_ego0 = load_snap_facebook_graph(ego_user=0)
    user = 0
    friends = set(g_ego0.neighbors(user))
    candidates = get_candidate_users(g_ego0, user)
    
    # User 0 is connected to all other 347 nodes in its ego network
    # For a different node in the ego network (e.g. node 1):
    test_node = 1
    recs = recommend_common_neighbors(g_ego0, user_id=test_node, top_k=5)
    assert len(recs) <= 5
    for r in recs:
        assert r["candidate_id"] != test_node
        assert not g_ego0.has_edge(test_node, r["candidate_id"])
