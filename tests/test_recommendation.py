"""Unit tests for Stage 2: Traditional Graph-based Recommendation Algorithms.
"""

import pytest
import networkx as nx

from src.data_loader import load_karate_club_graph
from src.recommendation import (
    get_candidate_users,
    get_mutual_friends,
    recommend_common_neighbors,
    recommend_jaccard,
    recommend_preferential_attachment,
    recommend_all_heuristics,
    compare_recommendations_dataframe,
)


@pytest.fixture
def karate_graph():
    """Fixture providing Zachary's Karate Club graph."""
    return load_karate_club_graph()


# ============================================================================
# Test Candidate Generation & Exclusions
# ============================================================================

def test_candidate_generation_exclusions(karate_graph):
    """Verify that candidates exclude query user and all existing friends."""
    user = 0
    friends = set(karate_graph.neighbors(user))
    candidates = get_candidate_users(karate_graph, user)
    
    # 1. Query user cannot be a candidate
    assert user not in candidates
    
    # 2. Existing friends cannot be candidates
    assert candidates.isdisjoint(friends)
    
    # 3. Candidate count must equal Total - 1 (self) - Degree
    expected_count = karate_graph.number_of_nodes() - 1 - len(friends)
    assert len(candidates) == expected_count


def test_invalid_user_id(karate_graph):
    """Verify ValueError is raised for non-existent user."""
    with pytest.raises(ValueError, match="does not exist"):
        get_candidate_users(karate_graph, user_id=999)


def test_invalid_top_k(karate_graph):
    """Verify ValueError is raised for non-positive top_k."""
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        recommend_common_neighbors(karate_graph, user_id=0, top_k=0)


# ============================================================================
# Test Common Neighbors
# ============================================================================

def test_common_neighbors_accuracy(karate_graph):
    """Verify Common Neighbors ranking and analytical score accuracy."""
    user = 0
    recs = recommend_common_neighbors(karate_graph, user, top_k=5)
    
    assert len(recs) == 5
    # For Node 0, Node 33 has 4 mutual friends (1, 2, 8, 31)
    top_rec = recs[0]
    assert top_rec["rank"] == 1
    assert top_rec["candidate_id"] == 33
    assert top_rec["score"] == 4
    assert set(top_rec["mutual_friends"]) == {8, 13, 19, 31}
    assert top_rec["num_mutual_friends"] == 4
    
    # Verify scores are monotonically non-increasing
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


# ============================================================================
# Test Jaccard Similarity
# ============================================================================

def test_jaccard_accuracy_and_cross_verification(karate_graph):
    """Verify Jaccard calculation matches manual formula and NetworkX."""
    user = 0
    recs = recommend_jaccard(karate_graph, user, top_k=5)
    
    deg_u = karate_graph.degree(user)
    for r in recs:
        candidate = r["candidate_id"]
        deg_v = karate_graph.degree(candidate)
        mutuals_count = r["num_mutual_friends"]
        expected_jaccard = round(mutuals_count / (deg_u + deg_v - mutuals_count), 4)
        
        assert abs(r["score"] - expected_jaccard) < 1e-4
        assert 0.0 <= r["score"] <= 1.0


# ============================================================================
# Test Preferential Attachment
# ============================================================================

def test_preferential_attachment_accuracy(karate_graph):
    """Verify Preferential Attachment score = deg(u) * deg(v)."""
    user = 0
    deg_u = karate_graph.degree(user)
    recs = recommend_preferential_attachment(karate_graph, user, top_k=5)
    
    for r in recs:
        deg_v = karate_graph.degree(r["candidate_id"])
        assert r["score"] == deg_u * deg_v
        
    # Top candidate by PA for user 0 must be Node 33 (degree 17 -> 16 * 17 = 272)
    assert recs[0]["candidate_id"] == 33
    assert recs[0]["score"] == 16 * 17


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_isolated_node_handling():
    """Verify recommendation behavior on a graph containing an isolated node."""
    g = nx.Graph()
    g.add_edge(1, 2)
    g.add_node(99)  # Isolated node
    
    # Recommendations for isolated node
    recs_cn = recommend_common_neighbors(g, user_id=99, top_k=2)
    assert len(recs_cn) == 2
    assert all(r["score"] == 0 for r in recs_cn)
    
    recs_jac = recommend_jaccard(g, user_id=99, top_k=2)
    assert len(recs_jac) == 2
    assert all(r["score"] == 0.0 for r in recs_jac)


def test_fully_connected_user_handling():
    """Verify handling when user is already connected to every node (complete graph)."""
    k4 = nx.complete_graph(4)
    recs = recommend_common_neighbors(k4, user_id=0, top_k=5)
    # No available candidates
    assert recs == []


def test_compare_recommendations_dataframe(karate_graph):
    """Verify dataframe comparison output."""
    df = compare_recommendations_dataframe(karate_graph, user_id=0, top_k=5)
    assert len(df) == 15  # 3 algorithms * 5 recommendations
    assert set(df["Algorithm"].unique()) == {"common_neighbors", "jaccard", "preferential_attachment"}
    assert "Mutual Friends" in df.columns
