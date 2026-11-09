"""Unit tests for Stage 1: Data Loading, Graph Construction, and Graph Analysis.
"""

from pathlib import Path
import pytest
import networkx as nx

from src.data_loader import (
    load_karate_club_graph,
    export_graph_to_edge_list_csv,
    load_edge_list_from_csv,
)
from src.graph_builder import (
    build_undirected_graph,
    validate_graph,
)
from src.graph_analysis import (
    compute_network_statistics,
    get_degree_distribution,
    get_clustering_coefficients,
    visualize_network,
)


# ============================================================================
# Test Data Loader
# ============================================================================

def test_load_karate_club_graph():
    """Verify that Karate Club loads with exact expected nodes and edges."""
    g = load_karate_club_graph()
    assert isinstance(g, nx.Graph)
    assert not g.is_directed()
    assert g.number_of_nodes() == 34
    assert g.number_of_edges() == 78
    
    # Check node attribute existence
    club_attrs = nx.get_node_attributes(g, "club")
    assert len(club_attrs) == 34
    assert set(club_attrs.values()) == {"Mr. Hi", "Officer"}


def test_export_and_load_edge_list_csv(tmp_path):
    """Verify round-trip export and reload of edge lists."""
    g = load_karate_club_graph()
    csv_file = tmp_path / "test_edges.csv"
    
    export_graph_to_edge_list_csv(g, csv_file)
    assert csv_file.is_file()
    
    loaded_edges = load_edge_list_from_csv(csv_file)
    assert len(loaded_edges) == 78
    assert all(len(edge) == 2 for edge in loaded_edges)


# ============================================================================
# Test Graph Builder
# ============================================================================

def test_build_undirected_graph_basic():
    """Verify simple undirected graph construction."""
    edges = [(1, 2), (2, 3), (3, 1)]
    g = build_undirected_graph(edges)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 3
    assert not g.is_directed()


def test_build_undirected_graph_deduplication():
    """Verify that reverse edges and duplicate edges are properly deduplicated."""
    edges = [(1, 2), (2, 1), (1, 2), (2, 3)]
    g = build_undirected_graph(edges)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2
    assert g.has_edge(1, 2)
    assert g.has_edge(2, 3)


def test_build_undirected_graph_self_loops_handling():
    """Verify that self-loops are filtered out by default."""
    edges = [(1, 2), (1, 1), (2, 2)]
    g = build_undirected_graph(edges, allow_self_loops=False)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    assert not g.has_edge(1, 1)


def test_build_undirected_graph_invalid_input():
    """Verify proper exceptions on empty or invalid edge input."""
    with pytest.raises(ValueError, match="empty"):
        build_undirected_graph([])
        
    with pytest.raises(ValueError, match="Malformed"):
        build_undirected_graph([(1,)])


def test_validate_graph_success():
    """Verify validation passes for valid Karate Club graph."""
    g = load_karate_club_graph()
    report = validate_graph(g)
    assert report["is_valid"] is True
    assert report["num_nodes"] == 34
    assert report["num_edges"] == 78
    assert report["num_self_loops"] == 0
    assert report["is_connected"] is True


def test_validate_graph_failure_on_empty():
    """Verify validation fails on empty graph."""
    g = nx.Graph()
    with pytest.raises(ValueError, match="0 nodes"):
        validate_graph(g)


# ============================================================================
# Test Graph Analysis
# ============================================================================

def test_compute_network_statistics_triangle():
    """Verify exact analytical values on a triangle graph K_3."""
    k3 = nx.complete_graph(3)
    stats = compute_network_statistics(k3)
    assert stats["num_nodes"] == 3
    assert stats["num_edges"] == 3
    assert stats["density"] == 1.0
    assert stats["average_degree"] == 2.0
    assert stats["is_connected"] is True
    assert stats["num_connected_components"] == 1
    assert stats["average_clustering_coefficient"] == 1.0


def test_compute_network_statistics_karate():
    """Verify statistical metrics on Zachary's Karate Club."""
    g = load_karate_club_graph()
    stats = compute_network_statistics(g)
    
    assert stats["num_nodes"] == 34
    assert stats["num_edges"] == 78
    # Average degree = 2 * 78 / 34 = 4.5882
    assert abs(stats["average_degree"] - 4.5882) < 0.001
    # Density = 2 * 78 / (34 * 33) = 156 / 1122 = 0.1390
    assert abs(stats["density"] - 0.1390) < 0.001
    assert stats["is_connected"] is True
    assert stats["num_connected_components"] == 1
    assert stats["average_clustering_coefficient"] > 0.5


def test_degree_and_clustering_distributions():
    """Verify distribution accessor functions."""
    g = load_karate_club_graph()
    degrees = get_degree_distribution(g)
    clusterings = get_clustering_coefficients(g)
    
    assert len(degrees) == 34
    assert degrees[0] == 16  # Node 0 (Mr. Hi) has degree 16
    assert degrees[33] == 17  # Node 33 (Officer) has degree 17
    assert len(clusterings) == 34
    assert all(0.0 <= c <= 1.0 for c in clusterings.values())


def test_visualize_network(tmp_path):
    """Verify that network visualization generates a valid image file."""
    g = load_karate_club_graph()
    img_path = tmp_path / "test_karate.png"
    result_path = visualize_network(g, output_path=img_path)
    
    assert result_path.is_file()
    assert result_path.stat().st_size > 1000  # File should not be empty
