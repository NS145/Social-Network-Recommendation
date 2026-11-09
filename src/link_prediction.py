"""Supervised link prediction module.
Implements leak-free train/test edge splitting, negative edge sampling,
node-pair feature engineering (Hadamard, Average, Absolute Difference),
Logistic Regression training, and calibrated probability recommendation ranking.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import random
import pickle
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression

from src.recommendation import get_candidate_users, get_mutual_friends
from src.embeddings import get_node_embedding


def split_edges(
    graph: nx.Graph,
    test_ratio: float = 0.2,
    seed: int = 42,
    ensure_connected: bool = True
) -> Tuple[nx.Graph, List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
    """Split graph edges into training and testing sets without data leakage.
    
    Data Leakage Prevention:
    1. Test edges are completely removed from the training graph G_train.
    2. Node representations (embeddings) must be learned ONLY on G_train.
    3. If ensure_connected is True, bridge edges whose removal would disconnect
       the graph are preserved in G_train, guaranteeing all nodes remain reachable.
       
    Args:
        graph: Original undirected NetworkX graph.
        test_ratio: Fraction of positive edges to reserve for testing.
        seed: Random seed for reproducible shuffling.
        ensure_connected: Whether to prevent network disconnection.
        
    Returns:
        Tuple containing:
        - g_train (nx.Graph): Training graph containing only training edges.
        - train_pos_edges (List[Tuple]): List of positive training edges.
        - test_pos_edges (List[Tuple]): List of positive test edges.
        
    Raises:
        TypeError: If graph is not a networkx.Graph.
        ValueError: If test_ratio is not in (0.0, 1.0).
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected networkx.Graph, got {type(graph).__name__}")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be strictly between 0.0 and 1.0")
        
    rng = random.Random(seed)
    
    # Work on a copy
    g_train = graph.copy()
    all_edges = list(graph.edges())
    rng.shuffle(all_edges)
    
    target_test_count = int(len(all_edges) * test_ratio)
    test_pos_edges: List[Tuple[Any, Any]] = []
    
    for u, v in all_edges:
        if len(test_pos_edges) >= target_test_count:
            break
            
        g_train.remove_edge(u, v)
        if ensure_connected:
            # Check if graph remains connected (or preserves initial components count)
            if nx.is_connected(g_train):
                test_pos_edges.append((u, v))
            else:
                # Removal broke connectivity; backtrack and keep edge in train
                g_train.add_edge(u, v)
        else:
            test_pos_edges.append((u, v))
            
    train_pos_edges = list(g_train.edges())
    
    # Assert zero overlap
    train_set = {tuple(sorted(e)) for e in train_pos_edges}
    test_set = {tuple(sorted(e)) for e in test_pos_edges}
    assert train_set.isdisjoint(test_set), "Data leakage detected: train and test positive edges overlap!"
    
    return g_train, train_pos_edges, test_pos_edges


def sample_negative_edges(
    graph: nx.Graph,
    num_train_neg: int,
    num_test_neg: int,
    seed: int = 42
) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
    """Sample negative edge examples (non-existent links) without data leakage.
    
    Negative Sampling Rules:
    1. Negative edges are sampled strictly from pairs (u, v) where no edge exists in the original graph.
    2. Training negative edges and test negative edges are completely disjoint.
    3. Self-loops (u == v) are excluded.
    
    Args:
        graph: Original undirected NetworkX graph.
        num_train_neg: Number of negative samples for training.
        num_test_neg: Number of negative samples for testing.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple containing:
        - train_neg_edges (List[Tuple]): Negative training edges.
        - test_neg_edges (List[Tuple]): Negative test edges.
        
    Raises:
        ValueError: If requested negative edges exceed total non-existent pairs.
    """
    nodes = sorted(list(graph.nodes()))
    total_nodes = len(nodes)
    
    # Collect all non-existent pairs (u < v)
    non_edges: List[Tuple[Any, Any]] = []
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            u, v = nodes[i], nodes[j]
            if not graph.has_edge(u, v):
                non_edges.append((u, v))
                
    total_required = num_train_neg + num_test_neg
    if total_required > len(non_edges):
        raise ValueError(
            f"Requested {total_required} negative edges, but graph only has "
            f"{len(non_edges)} non-existent edges."
        )
        
    rng = random.Random(seed)
    rng.shuffle(non_edges)
    
    train_neg_edges = non_edges[:num_train_neg]
    test_neg_edges = non_edges[num_train_neg:total_required]
    
    # Assert integrity: disjoint sets and no overlap with original edges
    train_neg_set = {tuple(sorted(e)) for e in train_neg_edges}
    test_neg_set = {tuple(sorted(e)) for e in test_neg_edges}
    orig_set = {tuple(sorted(e)) for e in graph.edges()}
    
    assert train_neg_set.isdisjoint(test_neg_set), "Overlap between train and test negative edges!"
    assert train_neg_set.isdisjoint(orig_set), "Negative training edge exists in true graph!"
    assert test_neg_set.isdisjoint(orig_set), "Negative test edge exists in true graph!"
    
    return train_neg_edges, test_neg_edges


def compute_edge_features(
    embeddings: Dict[Any, np.ndarray],
    u: Any,
    v: Any,
    operator: str = "hadamard"
) -> np.ndarray:
    """Construct a feature vector for a pair of nodes (u, v) from their embeddings.
    
    Feature Operators:
    - hadamard: Element-wise product: z_u * z_v (symmetric: f(u, v) == f(v, u))
    - average: Element-wise mean: (z_u + z_v) / 2 (symmetric)
    - abs_diff: Element-wise absolute difference: |z_u - z_v| (symmetric)
    - concat: Vector concatenation: [z_u, z_v]
    
    Args:
        embeddings: Dict mapping node ID to 1D embedding array.
        u: First node ID.
        v: Second node ID.
        operator: Feature composition operator ('hadamard', 'average', 'abs_diff', 'concat').
        
    Returns:
        np.ndarray: 1D edge feature vector.
        
    Raises:
        ValueError: If operator is unrecognized.
    """
    vec_u = get_node_embedding(embeddings, u)
    vec_v = get_node_embedding(embeddings, v)
    
    op = operator.lower()
    if op == "hadamard":
        return vec_u * vec_v
    elif op == "average":
        return (vec_u + vec_v) / 2.0
    elif op == "abs_diff":
        return np.abs(vec_u - vec_v)
    elif op == "concat":
        return np.concatenate([vec_u, vec_v])
    else:
        raise ValueError(
            f"Unknown edge operator '{operator}'. Choose from: 'hadamard', 'average', 'abs_diff', 'concat'."
        )


def build_link_prediction_dataset(
    pos_edges: List[Tuple[Any, Any]],
    neg_edges: List[Tuple[Any, Any]],
    embeddings: Dict[Any, np.ndarray],
    operator: str = "hadamard"
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct feature matrix X and label vector y for edge classification.
    
    Args:
        pos_edges: Positive edges (label 1).
        neg_edges: Negative edges (label 0).
        embeddings: Node embeddings dictionary.
        operator: Edge feature operator.
        
    Returns:
        Tuple (X, y) as NumPy arrays.
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    
    for u, v in pos_edges:
        feat = compute_edge_features(embeddings, u, v, operator=operator)
        X_list.append(feat)
        y_list.append(1)
        
    for u, v in neg_edges:
        feat = compute_edge_features(embeddings, u, v, operator=operator)
        X_list.append(feat)
        y_list.append(0)
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def train_link_prediction_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 42
) -> LogisticRegression:
    """Train a Logistic Regression classifier for link prediction.
    
    Args:
        X_train: Training feature matrix.
        y_train: Binary training labels (1 = link, 0 = no link).
        C: Inverse regularization strength.
        max_iter: Maximum solver iterations.
        seed: Random seed for reproducible solver initialization.
        
    Returns:
        LogisticRegression: Trained scikit-learn classifier.
    """
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    return clf


def save_link_prediction_model(
    model: LogisticRegression,
    metadata: Dict[str, Any],
    output_path: str | Path
) -> Path:
    """Serialize the trained link prediction classifier and metadata to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "metadata": metadata
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_link_prediction_model(file_path: str | Path) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Load a serialized link prediction model and its configuration metadata."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["metadata"]


def recommend_supervised_link_prediction(
    graph: nx.Graph,
    model: LogisticRegression,
    embeddings: Dict[Any, np.ndarray],
    user_id: Any,
    top_k: int = 5,
    operator: str = "hadamard"
) -> List[Dict[str, Any]]:
    """Generate friend recommendations using supervised link prediction probabilities.
    
    Exclusion Rules:
    - The query user themself is excluded.
    - All existing friends in graph are excluded.
    
    Args:
        graph: Original undirected NetworkX graph.
        model: Trained LogisticRegression classifier.
        embeddings: Node embeddings dictionary.
        user_id: Query node ID.
        top_k: Number of recommendations to return.
        operator: Edge feature operator used during training.
        
    Returns:
        List of recommendation dictionaries sorted descending by predicted probability.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    candidates = get_candidate_users(graph, user_id)
    if not candidates:
        return []
        
    cand_list = list(candidates)
    features = [
        compute_edge_features(embeddings, user_id, cand, operator=operator)
        for cand in cand_list
    ]
    X_cand = np.array(features, dtype=np.float32)
    
    # Predict calibrated probability of link existence (class 1)
    probabilities = model.predict_proba(X_cand)[:, 1]
    
    scored_candidates = []
    for cand, prob in zip(cand_list, probabilities):
        mutuals = get_mutual_friends(graph, user_id, cand)
        scored_candidates.append({
            "candidate_id": cand,
            "score": round(float(prob), 4),
            "mutual_friends": mutuals,
            "num_mutual_friends": len(mutuals),
            "candidate_degree": graph.degree(cand),
        })
        
    # Rank descending by predicted probability, tie-break by candidate_id
    scored_candidates.sort(key=lambda x: (-x["score"], str(x["candidate_id"])))
    
    top_recommendations = scored_candidates[:top_k]
    for rank, item in enumerate(top_recommendations, start=1):
        item["rank"] = rank
        
    return top_recommendations
