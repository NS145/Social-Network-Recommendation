"""Unit tests for Stage 6: Streamlit Application Components and UI Pipeline.
"""

import pytest
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

from app import (
    get_cached_graph_and_stats,
    get_cached_models_and_embeddings,
    get_cached_evaluation_dataframe,
)
from src.graph_analysis import create_interactive_network_figure
from src.recommendation import recommend_all_heuristics, get_candidate_users
from src.embeddings import recommend_node2vec_cosine
from src.link_prediction import recommend_supervised_link_prediction


# ============================================================================
# Test Cached Data & Model Loaders
# ============================================================================

def test_get_cached_graph_and_stats():
    """Verify graph loader returns valid Karate Club graph and stats."""
    graph, stats = get_cached_graph_and_stats()
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 34
    assert graph.number_of_edges() == 78
    assert "num_nodes" in stats
    assert "density" in stats
    assert stats["num_nodes"] == 34


def test_get_cached_models_and_embeddings():
    """Verify model loader returns non-empty embeddings and trained classifier."""
    graph, _ = get_cached_graph_and_stats()
    embeddings, clf, metadata = get_cached_models_and_embeddings(graph)
    
    assert len(embeddings) == 34
    assert isinstance(clf, LogisticRegression)
    assert "operator" in metadata
    assert metadata["operator"] == "hadamard"


def test_get_cached_evaluation_dataframe():
    """Verify evaluation benchmark DataFrame generation."""
    graph, _ = get_cached_graph_and_stats()
    embeddings, clf, _ = get_cached_models_and_embeddings(graph)
    df_eval = get_cached_evaluation_dataframe(graph, embeddings, clf, k=5)
    
    assert isinstance(df_eval, pd.DataFrame)
    assert len(df_eval) == 5
    assert "ROC-AUC" in df_eval.columns
    assert "Precision@5" in df_eval.columns


# ============================================================================
# Test Interactive Plotly Network Figure
# ============================================================================

def test_create_interactive_network_figure():
    """Verify Plotly figure generation with selected user and recommendations."""
    graph, _ = get_cached_graph_and_stats()
    fig = create_interactive_network_figure(
        graph,
        selected_user=0,
        recommended_candidates=[16, 9, 28]
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # Figure should contain edge lines and node marker traces
    trace_names = [trace.name for trace in fig.data if trace.name is not None]
    assert "Existing Friendships" in trace_names
    assert "Predicted Link" in trace_names
    assert "Selected User" in trace_names
    assert "Recommended Friends" in trace_names


# ============================================================================
# Test Recommendation Pipeline Integration for UI
# ============================================================================

def test_all_algorithms_ui_pipeline():
    """Verify all 5 algorithms successfully produce top-k recommendations for UI."""
    graph, _ = get_cached_graph_and_stats()
    embeddings, clf, _ = get_cached_models_and_embeddings(graph)
    
    user = 0
    top_k = 5
    candidates = get_candidate_users(graph, user)
    friends = set(graph.neighbors(user))
    
    heuristics = recommend_all_heuristics(graph, user, top_k=top_k)
    n2v_recs = recommend_node2vec_cosine(graph, embeddings, user, top_k=top_k)
    sup_recs = recommend_supervised_link_prediction(graph, clf, embeddings, user, top_k=top_k)
    
    for algo_name, rec_list in [
        ("Common Neighbors", heuristics["common_neighbors"]),
        ("Jaccard", heuristics["jaccard"]),
        ("Preferential Attachment", heuristics["preferential_attachment"]),
        ("Node2Vec Cosine", n2v_recs),
        ("Supervised LR", sup_recs),
    ]:
        assert len(rec_list) == top_k, f"{algo_name} returned {len(rec_list)} items"
        for item in rec_list:
            cand = item["candidate_id"]
            assert cand != user, f"{algo_name} recommended self!"
            assert cand not in friends, f"{algo_name} recommended existing friend {cand}!"
            assert cand in candidates, f"{algo_name} recommended non-candidate {cand}!"


def test_streamlit_app_renders_without_exceptions():
    """Verify that the full Streamlit app runs without unhandled exceptions."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert len(at.exception) == 0, f"App threw exceptions: {at.exception}"

