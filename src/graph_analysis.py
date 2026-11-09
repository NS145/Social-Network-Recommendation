"""Graph analysis and visualization module.
Computes comprehensive topological statistics (degree distributions, density,
connected components, clustering coefficients) and produces reproducible network plots.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def compute_network_statistics(graph: nx.Graph) -> Dict[str, Any]:
    """Calculate core structural properties of an undirected network.
    
    Mathematical metrics computed:
    - Number of Nodes (|V|)
    - Number of Edges (|E|)
    - Graph Density: rho = 2 * |E| / (|V| * (|V| - 1))
    - Average Degree: <k> = 2 * |E| / |V|
    - Degree Summary: min, max, median degree
    - Connected Components: total count, size of giant component
    - Average Local Clustering Coefficient: C = (1 / |V|) * sum(C_v)
    - Transitivity (Global Clustering): ratio of 3 * triangles / triads
    
    Args:
        graph: Undirected NetworkX graph.
        
    Returns:
        dict: Key-value dictionary of network metrics.
        
    Raises:
        TypeError: If graph is not a networkx.Graph.
        ValueError: If graph is empty.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected networkx.Graph, got {type(graph).__name__}")
        
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        raise ValueError("Cannot compute statistics for an empty graph.")
        
    num_edges = graph.number_of_edges()
    
    # Degrees
    degrees = dict(graph.degree())
    deg_values = list(degrees.values())
    avg_degree = float(np.mean(deg_values)) if num_nodes > 0 else 0.0
    min_degree = int(np.min(deg_values)) if num_nodes > 0 else 0
    max_degree = int(np.max(deg_values)) if num_nodes > 0 else 0
    median_degree = float(np.median(deg_values)) if num_nodes > 0 else 0.0
    
    # Density: rho = 2 * |E| / (|V| * (|V| - 1))
    density = nx.density(graph)
    
    # Connected components
    connected_components = list(nx.connected_components(graph))
    num_connected_components = len(connected_components)
    is_connected = nx.is_connected(graph)
    largest_cc_size = max(len(c) for c in connected_components) if connected_components else 0
    
    # Clustering coefficients
    clustering_dict = nx.clustering(graph)
    avg_clustering = float(nx.average_clustering(graph))
    transitivity = float(nx.transitivity(graph))
    
    # Shortest paths (computed on largest connected component if disconnected)
    if is_connected:
        diameter = nx.diameter(graph)
        avg_shortest_path = float(nx.average_shortest_path_length(graph))
    else:
        giant_subgraph = graph.subgraph(connected_components[0]).copy()
        diameter = nx.diameter(giant_subgraph)
        avg_shortest_path = float(nx.average_shortest_path_length(giant_subgraph))
        
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": round(density, 4),
        "average_degree": round(avg_degree, 4),
        "min_degree": min_degree,
        "max_degree": max_degree,
        "median_degree": round(median_degree, 2),
        "is_connected": is_connected,
        "num_connected_components": num_connected_components,
        "largest_cc_size": largest_cc_size,
        "average_clustering_coefficient": round(avg_clustering, 4),
        "transitivity": round(transitivity, 4),
        "diameter": diameter,
        "average_shortest_path_length": round(avg_shortest_path, 4),
    }


def get_degree_distribution(graph: nx.Graph) -> Dict[Any, int]:
    """Return a mapping of node identifier to its graph degree."""
    return dict(graph.degree())


def get_clustering_coefficients(graph: nx.Graph) -> Dict[Any, float]:
    """Return a mapping of node identifier to its local clustering coefficient."""
    return {node: round(score, 4) for node, score in nx.clustering(graph).items()}


def visualize_network(
    graph: nx.Graph,
    output_path: Optional[str | Path] = None,
    title: str = "Social Network Graph",
    seed: int = 42,
    node_color_attr: Optional[str] = "club"
) -> Path:
    """Render and save an aesthetically formatted network diagram.
    
    Args:
        graph: Undirected NetworkX graph.
        output_path: Destination path for saved figure (PNG).
        title: Plot title.
        seed: Random seed for reproducible spring layout.
        node_color_attr: Node attribute to map colors to (e.g. 'club').
        
    Returns:
        Path: Path to the saved image file.
    """
    if output_path is None:
        output_path = Path("reports") / "figures" / "karate_club_network.png"
    else:
        output_path = Path(output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Reproducible layout
    pos = nx.spring_layout(graph, seed=seed, k=0.35, iterations=50)
    
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Determine node colors based on community/club attribute if present
    node_colors = []
    if node_color_attr and any(node_color_attr in data for _, data in graph.nodes(data=True)):
        for _, data in graph.nodes(data=True):
            val = data.get(node_color_attr, "Unknown")
            if val == "Mr. Hi":
                node_colors.append("#1f77b4")  # Blue
            elif val == "Officer":
                node_colors.append("#ff7f0e")  # Orange
            else:
                node_colors.append("#2ca02c")  # Green
    else:
        node_colors = ["#1f77b4"] * graph.number_of_nodes()
        
    # Scale node size by degree for visual salience
    degrees = dict(graph.degree())
    node_sizes = [300 + 70 * degrees[node] for node in graph.nodes()]
    
    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos,
        alpha=0.35,
        edge_color="#7f7f7f",
        width=1.2
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#222222",
        linewidths=1.2,
        alpha=0.9
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=9,
        font_family="sans-serif",
        font_color="#ffffff",
        font_weight="bold"
    )
    
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    return output_path
