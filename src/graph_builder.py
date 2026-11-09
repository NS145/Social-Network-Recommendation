"""Graph construction and validation module.
Responsible for building NetworkX graphs, validating node identifiers,
handling edge consistency, and ensuring undirected simple graph semantics.
"""

from typing import Iterable, Tuple, Dict, Any, Optional, Set
import networkx as nx


def build_undirected_graph(
    edges: Iterable[Tuple[Any, Any]],
    node_attributes: Optional[Dict[Any, Dict[str, Any]]] = None,
    allow_self_loops: bool = False
) -> nx.Graph:
    """Build a validated, undirected simple graph from an edge collection.
    
    In social network analysis, a friendship is symmetric and undirected:
    if Alice is friends with Bob, Bob is friends with Alice.
    Self-loops (Alice friends with Alice) and multi-edges are invalid.
    
    Args:
        edges: Iterable of (u, v) pairs representing friendships.
        node_attributes: Optional dictionary mapping node_id -> {attr_key: attr_value}.
        allow_self_loops: Whether self-loops (u == v) are permitted. Default False.
        
    Returns:
        nx.Graph: A clean, undirected NetworkX Graph.
        
    Raises:
        ValueError: If edges is empty or contains malformed pairs.
    """
    if edges is None:
        raise ValueError("Edges collection cannot be None.")
        
    edge_list = list(edges)
    if len(edge_list) == 0:
        raise ValueError("Cannot build graph: edges collection is empty.")
        
    graph = nx.Graph()
    
    seen_edges: Set[Tuple[Any, Any]] = set()
    self_loops_count = 0
    
    for idx, edge in enumerate(edge_list):
        if not (isinstance(edge, (tuple, list)) and len(edge) >= 2):
            raise ValueError(f"Malformed edge at index {idx}: expected tuple of length >= 2, got {edge}")
            
        u, v = edge[0], edge[1]
        
        # Check self-loops
        if u == v:
            self_loops_count += 1
            if not allow_self_loops:
                continue
                
        # Canonical undirected edge representation
        canonical_edge = (min(u, v), max(u, v)) if isinstance(u, type(v)) and (type(u) in (int, float, str)) else (u, v)
        if canonical_edge in seen_edges:
            continue
            
        seen_edges.add(canonical_edge)
        graph.add_edge(u, v)
        
    # Attach node attributes if provided
    if node_attributes:
        for node, attrs in node_attributes.items():
            if graph.has_node(node):
                for k, v in attrs.items():
                    graph.nodes[node][k] = v
            else:
                # Add isolated node with attributes
                graph.add_node(node, **attrs)
                
    return graph


def validate_graph(graph: nx.Graph) -> Dict[str, Any]:
    """Validate graph integrity and topological properties.
    
    Args:
        graph: NetworkX Graph to inspect.
        
    Returns:
        dict: Summary of validation checks including:
            - is_valid (bool)
            - is_directed (bool)
            - num_nodes (int)
            - num_edges (int)
            - num_self_loops (int)
            - is_connected (bool)
            - has_isolated_nodes (bool)
            
    Raises:
        TypeError: If graph is not a networkx.Graph instance.
        ValueError: If graph contains 0 nodes.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected networkx.Graph, got {type(graph).__name__}")
        
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        raise ValueError("Graph validation failed: graph contains 0 nodes.")
        
    num_self_loops = nx.number_of_selfloops(graph)
    is_directed = graph.is_directed()
    num_edges = graph.number_of_edges()
    is_connected = nx.is_connected(graph) if num_nodes > 0 else False
    num_isolated = len(list(nx.isolates(graph)))
    
    is_valid = (
        not is_directed
        and num_nodes > 0
        and num_self_loops == 0
    )
    
    return {
        "is_valid": is_valid,
        "is_directed": is_directed,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_self_loops": num_self_loops,
        "is_connected": is_connected,
        "num_isolated_nodes": num_isolated,
    }
