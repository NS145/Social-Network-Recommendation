"""Dataset loading module.
Responsible for acquiring, loading, and serializing raw graph datasets.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import networkx as nx
import pandas as pd


def load_karate_club_graph() -> nx.Graph:
    """Load Zachary's Karate Club graph directly from NetworkX.
    
    Zachary's Karate Club is a well-known social network of a university
    karate club described by Wayne Zachary in 1977. It contains 34 members
    (nodes 0 to 33) and 78 pairwise friendships (edges).
    
    Returns:
        nx.Graph: Undirected NetworkX Graph with 34 nodes, 78 edges,
                  and a 'club' attribute on each node ('Mr. Hi' or 'Officer').
    """
    graph = nx.karate_club_graph()
    return graph


def export_graph_to_edge_list_csv(
    graph: nx.Graph,
    output_path: str | Path,
    include_header: bool = True
) -> Path:
    """Export graph edges to a CSV file in edge-list format.
    
    Args:
        graph: Undirected NetworkX graph.
        output_path: Destination path for the CSV.
        include_header: Whether to write 'source,target' header.
        
    Returns:
        Path: Absolute path to the saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    edges = list(graph.edges())
    df = pd.DataFrame(edges, columns=["source", "target"])
    df.to_csv(path, index=False, header=include_header)
    return path


def load_edge_list_from_csv(
    file_path: str | Path,
    source_col: str = "source",
    target_col: str = "target",
    delimiter: str = ","
) -> List[Tuple[int, int]]:
    """Load an edge list from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        source_col: Name of source node column.
        target_col: Name of target node column.
        delimiter: CSV separator.
        
    Returns:
        List of (source, target) tuples.
        
    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If specified columns are not in the CSV.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Edge list file not found: {file_path}")
        
    df = pd.read_csv(path, delimiter=delimiter)
    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"Columns '{source_col}' and '{target_col}' not found in {file_path}. "
            f"Found: {list(df.columns)}"
        )
        
    edges = [
        (int(row[source_col]), int(row[target_col]))
        for _, row in df[[source_col, target_col]].iterrows()
    ]
    return edges
