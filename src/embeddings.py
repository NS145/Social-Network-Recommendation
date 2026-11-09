"""Node representation learning module using Node2Vec.
Implements biased random walk generation, Skip-gram embedding training,
continuous representation persistence, and cosine similarity ranking.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pickle
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec

from src.recommendation import get_candidate_users, get_mutual_friends


def train_node2vec(
    graph: nx.Graph,
    dimensions: int = 32,
    walk_length: int = 10,
    num_walks: int = 20,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 5,
    min_count: int = 1,
    seed: int = 42,
    workers: int = 1,
    quiet: bool = True
) -> Word2Vec:
    """Train a Node2Vec model on an undirected graph.
    
    Node2Vec maps nodes to continuous feature representations by optimizing a
    neighborhood-preserving objective via biased 2nd-order random walks and
    Word2Vec Skip-gram architecture.
    
    Parameters:
        graph: Undirected NetworkX graph.
        dimensions: Dimensionality of learned node embedding vectors (d).
        walk_length: Number of nodes visited in each random walk (l).
        num_walks: Number of random walks generated per node (r).
        p: Return parameter. Controls probability of returning to previous node.
           High p (> 1) encourages exploration; Low p (< 1) keeps walks local.
        q: In-out parameter. Controls BFS vs DFS exploration.
           High q (> 1) biases towards BFS / homophily (community cluster).
           Low q (< 1) biases towards DFS / structural equivalence (roles).
        window: Maximum skip-gram context window size.
        min_count: Minimum occurrence threshold for node inclusion.
        seed: Random seed for reproducible walks and Word2Vec weight initialization.
        workers: CPU worker threads. Set to 1 for strict determinism on Windows.
        quiet: Suppress stdout progress bars.
        
    Returns:
        gensim.models.Word2Vec: Trained model containing learned node vectors.
        
    Raises:
        TypeError: If graph is not a networkx.Graph.
        ValueError: If graph is empty or parameters are invalid.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected networkx.Graph, got {type(graph).__name__}")
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot train Node2Vec on an empty graph.")
    if dimensions <= 0 or walk_length <= 0 or num_walks <= 0:
        raise ValueError("dimensions, walk_length, and num_walks must be positive integers.")
    if p <= 0 or q <= 0:
        raise ValueError("Hyperparameters p and q must be strictly positive.")
        
    # Instantiate Node2Vec random walk generator
    node2vec_walker = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
        quiet=quiet
    )
    
    # Fit Skip-gram Word2Vec model on generated walk sequences
    model = node2vec_walker.fit(
        window=window,
        min_count=min_count,
        seed=seed,
        workers=workers
    )
    
    return model


def get_node_embedding(
    model_or_embeddings: Union[Word2Vec, Dict[Any, np.ndarray]],
    node_id: Any,
    dimensions: Optional[int] = None
) -> np.ndarray:
    """Retrieve the continuous embedding vector for a given node.
    
    Handles type conversion between integer and string identifiers.
    
    Args:
        model_or_embeddings: Word2Vec model or dictionary of {node_id: np.ndarray}.
        node_id: Target node identifier (int or str).
        dimensions: Expected vector dimension for fallback zero-vector.
        
    Returns:
        np.ndarray: 1D embedding vector of shape (dimensions,).
    """
    if isinstance(model_or_embeddings, dict):
        if node_id in model_or_embeddings:
            return model_or_embeddings[node_id]
        str_id = str(node_id)
        if str_id in model_or_embeddings:
            return model_or_embeddings[str_id]
        if dimensions is None:
            # Infer dimension from first item
            dimensions = len(next(iter(model_or_embeddings.values())))
        return np.zeros(dimensions, dtype=np.float32)
        
    # Assume gensim Word2Vec model
    key = str(node_id)
    if key in model_or_embeddings.wv:
        return model_or_embeddings.wv[key]
    elif node_id in model_or_embeddings.wv:
        return model_or_embeddings.wv[node_id]
        
    dim = dimensions or model_or_embeddings.vector_size
    return np.zeros(dim, dtype=np.float32)


def extract_all_embeddings(
    model: Word2Vec,
    graph: nx.Graph
) -> Dict[Any, np.ndarray]:
    """Extract a dictionary mapping all graph nodes to their embedding arrays.
    
    Preserves original node types (e.g. integer keys).
    
    Args:
        model: Trained gensim Word2Vec model.
        graph: Target NetworkX graph.
        
    Returns:
        Dict[Any, np.ndarray]: Node ID -> 1D embedding vector.
    """
    embeddings = {}
    dim = model.vector_size
    for node in graph.nodes():
        embeddings[node] = get_node_embedding(model, node, dimensions=dim)
    return embeddings


def save_embeddings(
    embeddings_dict: Dict[Any, np.ndarray],
    output_path: str | Path
) -> Path:
    """Save an embeddings dictionary to disk via pickle.
    
    Args:
        embeddings_dict: Mapping of node ID to numpy embedding array.
        output_path: Target file destination (.pkl).
        
    Returns:
        Path: Absolute path to saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_embeddings(file_path: str | Path) -> Dict[Any, np.ndarray]:
    """Load serialized embeddings dictionary from disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {file_path}")
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def save_node2vec_model(model: Word2Vec, output_path: str | Path) -> Path:
    """Serialize the complete Word2Vec model to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path


def load_node2vec_model(file_path: str | Path) -> Word2Vec:
    """Load a saved Word2Vec model from disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    return Word2Vec.load(str(path))


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two continuous vectors.
    
    Formula:
        Cosine_Similarity(a, b) = (a · b) / (||a||_2 * ||b||_2)
        
    Returns:
        float: Cosine similarity score in [-1.0, 1.0].
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    # Clip for floating point stability
    return float(np.clip(similarity, -1.0, 1.0))


def recommend_node2vec_cosine(
    graph: nx.Graph,
    model_or_embeddings: Union[Word2Vec, Dict[Any, np.ndarray]],
    user_id: Any,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Rank friend recommendations based on Node2Vec embedding cosine similarity.
    
    Exclusion Rules:
    - The query user themself is excluded.
    - All existing friends are excluded.
    
    Args:
        graph: Undirected NetworkX graph.
        model_or_embeddings: Trained Word2Vec model or embeddings dict.
        user_id: Query node ID.
        top_k: Number of recommendations to return.
        
    Returns:
        List of recommendation dictionaries sorted descending by cosine similarity.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    candidates = get_candidate_users(graph, user_id)
    if not candidates:
        return []
        
    user_vec = get_node_embedding(model_or_embeddings, user_id)
    user_neighbors = set(graph.neighbors(user_id))
    
    scored_candidates = []
    for candidate in candidates:
        cand_vec = get_node_embedding(model_or_embeddings, candidate)
        score = compute_cosine_similarity(user_vec, cand_vec)
        mutuals = get_mutual_friends(graph, user_id, candidate)
        
        scored_candidates.append({
            "candidate_id": candidate,
            "score": round(score, 4),
            "mutual_friends": mutuals,
            "num_mutual_friends": len(mutuals),
            "candidate_degree": graph.degree(candidate),
        })
        
    # Rank descending by cosine similarity, tie-break by candidate_id
    scored_candidates.sort(key=lambda x: (-x["score"], str(x["candidate_id"])))
    
    top_recommendations = scored_candidates[:top_k]
    for rank, item in enumerate(top_recommendations, start=1):
        item["rank"] = rank
        
    return top_recommendations
