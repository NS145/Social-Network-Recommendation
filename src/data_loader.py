"""Dataset loading module.
Responsible for acquiring, loading, and serializing raw graph datasets.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Set
import urllib.request
import gzip
import shutil
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


def download_snap_facebook_dataset(
    dest_dir: str | Path = "data/raw",
    url: str = "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    timeout: int = 30
) -> Path:
    """Download and decompress Stanford SNAP Facebook Social Circles edge list.
    
    Source: J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego Networks.
    NIPS 2012. https://snap.stanford.edu/data/ego-Facebook.html
    
    Dataset Characteristics:
    - 4,039 anonymized nodes (survey participants and friends)
    - 88,234 undirected edges (mutual friendships)
    - 10 ego networks
    
    Args:
        dest_dir: Directory to store raw files.
        url: URL of the compressed edge-list file.
        timeout: Network timeout in seconds.
        
    Returns:
        Path: Path to decompressed edge list file.
    """
    raw_dir = Path(dest_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    gz_path = raw_dir / "facebook_combined.txt.gz"
    txt_path = raw_dir / "facebook_combined.txt"
    
    # Return immediately if file already exists
    if txt_path.is_file() and txt_path.stat().st_size > 0:
        return txt_path
        
    # Download compressed file
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(gz_path, "wb") as out_f:
        shutil.copyfileobj(resp, out_f)
        
    # Decompress
    with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
        
    return txt_path


def load_snap_facebook_graph(
    file_path: Optional[str | Path] = None,
    ego_user: Optional[int] = None,
    max_nodes: Optional[int] = None
) -> nx.Graph:
    """Load the Stanford SNAP Facebook network as an undirected NetworkX graph.
    
    Args:
        file_path: Optional path to decompressed facebook_combined.txt.
                   If None, looks in data/raw or downloads automatically.
        ego_user: If specified, extracts the induced ego-network subgraph around
                  this user (the user and all their 1-hop friends).
                  E.g., ego_user=0 yields 348 nodes and 2,866 edges.
        max_nodes: If specified, limits the graph to an induced subgraph of size max_nodes.
        
    Returns:
        nx.Graph: Clean, undirected simple graph.
    """
    if file_path is None:
        file_path = Path("data") / "raw" / "facebook_combined.txt"
        if not file_path.is_file():
            file_path = download_snap_facebook_dataset()
    else:
        file_path = Path(file_path)
        
    if not file_path.is_file():
        raise FileNotFoundError(f"SNAP Facebook file not found at: {file_path}")
        
    edges: List[Tuple[int, int]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:  # Filter self-loops
                    edges.append((u, v))
                    
    graph = nx.Graph(edges)
    
    # Filter by ego network if requested
    if ego_user is not None:
        if ego_user not in graph:
            raise ValueError(f"Ego user {ego_user} does not exist in Facebook graph.")
        ego_nodes = set(graph.neighbors(ego_user)) | {ego_user}
        subgraph = graph.subgraph(ego_nodes).copy()
        return subgraph
        
    # Filter by max nodes if requested
    if max_nodes is not None and max_nodes < graph.number_of_nodes():
        selected_nodes = sorted(list(graph.nodes()))[:max_nodes]
        return graph.subgraph(selected_nodes).copy()
        
    return graph
