"""Traditional graph-based recommendation module.
Implements neighborhood-based and degree-based heuristic link prediction algorithms:
- Common Neighbors (CN)
- Jaccard Coefficient (J)
- Preferential Attachment (PA)

All recommendation functions adhere to strict social network constraints:
1. Self-recommendations are excluded.
2. Existing friends are excluded.
3. Candidate recommendations are ranked descending by score with deterministic tie-breaking.
4. Mutual friend IDs and counts are included in results.
5. Isolated users and invalid user IDs are handled safely.
"""

from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import pandas as pd


def get_candidate_users(graph: nx.Graph, user_id: Any) -> Set[Any]:
    """Retrieve all eligible candidate nodes for recommendation.
    
    A candidate must:
    - Exist in the network.
    - Not be the query user themself (no self-recommendations).
    - Not already be connected to the query user (no existing friends).
    
    Args:
        graph: Undirected NetworkX graph.
        user_id: Target user node identifier.
        
    Returns:
        Set of valid candidate node identifiers.
        
    Raises:
        TypeError: If graph is not a networkx.Graph.
        ValueError: If user_id is not in graph.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected networkx.Graph, got {type(graph).__name__}")
    if user_id not in graph:
        raise ValueError(f"User ID '{user_id}' does not exist in the graph.")
        
    existing_friends = set(graph.neighbors(user_id))
    all_nodes = set(graph.nodes())
    candidates = all_nodes - existing_friends - {user_id}
    return candidates


def get_mutual_friends(graph: nx.Graph, u: Any, v: Any) -> List[Any]:
    """Return a sorted list of mutual friends (common neighbors) between u and v."""
    if u not in graph or v not in graph:
        return []
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    return sorted(list(neighbors_u.intersection(neighbors_v)))


def recommend_common_neighbors(
    graph: nx.Graph,
    user_id: Any,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Generate friend recommendations using Common Neighbors.
    
    Mathematical Definition:
        CN(u, v) = |N(u) ∩ N(v)|
        
    Intuition:
        Homophily & Triadic Closure: Individuals who share more mutual friends
        have higher social proximity and are more likely to establish a future link.
        
    Args:
        graph: Undirected NetworkX graph.
        user_id: Query node ID.
        top_k: Maximum number of recommendations to return.
        
    Returns:
        List of recommendation dictionaries sorted descending by score.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    candidates = get_candidate_users(graph, user_id)
    if not candidates:
        return []
        
    user_neighbors = set(graph.neighbors(user_id))
    scored_candidates = []
    
    for candidate in candidates:
        cand_neighbors = set(graph.neighbors(candidate))
        mutuals = sorted(list(user_neighbors.intersection(cand_neighbors)))
        score = len(mutuals)
        scored_candidates.append({
            "candidate_id": candidate,
            "score": score,
            "mutual_friends": mutuals,
            "num_mutual_friends": score,
            "candidate_degree": graph.degree(candidate),
        })
        
    # Rank descending by score, tie-break by candidate_id
    scored_candidates.sort(key=lambda x: (-x["score"], str(x["candidate_id"])))
    
    top_recommendations = scored_candidates[:top_k]
    for rank, item in enumerate(top_recommendations, start=1):
        item["rank"] = rank
        
    return top_recommendations


def recommend_jaccard(
    graph: nx.Graph,
    user_id: Any,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Generate friend recommendations using Jaccard Similarity Coefficient.
    
    Mathematical Definition:
        J(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
                = |N(u) ∩ N(v)| / (deg(u) + deg(v) - |N(u) ∩ N(v)|)
                
    Intuition:
        Normalizes Common Neighbors by the union size of their friend networks.
        This penalizes high-degree hub nodes and rewards candidates whose friend circle
        substantially overlaps with the query user.
        
    Args:
        graph: Undirected NetworkX graph.
        user_id: Query node ID.
        top_k: Maximum number of recommendations to return.
        
    Returns:
        List of recommendation dictionaries sorted descending by score.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    candidates = get_candidate_users(graph, user_id)
    if not candidates:
        return []
        
    user_neighbors = set(graph.neighbors(user_id))
    deg_u = len(user_neighbors)
    scored_candidates = []
    
    for candidate in candidates:
        cand_neighbors = set(graph.neighbors(candidate))
        deg_v = len(cand_neighbors)
        mutuals = sorted(list(user_neighbors.intersection(cand_neighbors)))
        num_mutual = len(mutuals)
        union_size = deg_u + deg_v - num_mutual
        
        score = (num_mutual / union_size) if union_size > 0 else 0.0
        
        scored_candidates.append({
            "candidate_id": candidate,
            "score": round(score, 4),
            "mutual_friends": mutuals,
            "num_mutual_friends": num_mutual,
            "candidate_degree": deg_v,
        })
        
    # Rank descending by score, tie-break by candidate_id
    scored_candidates.sort(key=lambda x: (-x["score"], str(x["candidate_id"])))
    
    top_recommendations = scored_candidates[:top_k]
    for rank, item in enumerate(top_recommendations, start=1):
        item["rank"] = rank
        
    return top_recommendations


def recommend_preferential_attachment(
    graph: nx.Graph,
    user_id: Any,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Generate friend recommendations using Preferential Attachment.
    
    Mathematical Definition:
        PA(u, v) = deg(u) × deg(v)
        
    Intuition:
        Scale-free "rich-get-richer" dynamics (Barabási-Albert model).
        Highly connected nodes are more likely to attract new connections.
        Note: For a fixed user u, deg(u) is constant, meaning ranking by PA(u, v)
        is equivalent to ranking candidates strictly by their global degree deg(v).
        
    Args:
        graph: Undirected NetworkX graph.
        user_id: Query node ID.
        top_k: Maximum number of recommendations to return.
        
    Returns:
        List of recommendation dictionaries sorted descending by score.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    candidates = get_candidate_users(graph, user_id)
    if not candidates:
        return []
        
    user_neighbors = set(graph.neighbors(user_id))
    deg_u = len(user_neighbors)
    scored_candidates = []
    
    for candidate in candidates:
        cand_neighbors = set(graph.neighbors(candidate))
        deg_v = len(cand_neighbors)
        mutuals = sorted(list(user_neighbors.intersection(cand_neighbors)))
        score = deg_u * deg_v
        
        scored_candidates.append({
            "candidate_id": candidate,
            "score": score,
            "mutual_friends": mutuals,
            "num_mutual_friends": len(mutuals),
            "candidate_degree": deg_v,
        })
        
    # Rank descending by score, tie-break by candidate_id
    scored_candidates.sort(key=lambda x: (-x["score"], str(x["candidate_id"])))
    
    top_recommendations = scored_candidates[:top_k]
    for rank, item in enumerate(top_recommendations, start=1):
        item["rank"] = rank
        
    return top_recommendations


def recommend_all_heuristics(
    graph: nx.Graph,
    user_id: Any,
    top_k: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute recommendations across all three traditional heuristics simultaneously."""
    return {
        "common_neighbors": recommend_common_neighbors(graph, user_id, top_k),
        "jaccard": recommend_jaccard(graph, user_id, top_k),
        "preferential_attachment": recommend_preferential_attachment(graph, user_id, top_k),
    }


def compare_recommendations_dataframe(
    graph: nx.Graph,
    user_id: Any,
    top_k: int = 5
) -> pd.DataFrame:
    """Produce a consolidated comparison table across all traditional algorithms.
    
    Args:
        graph: Undirected NetworkX graph.
        user_id: Query node ID.
        top_k: Top-k elements per algorithm.
        
    Returns:
        pd.DataFrame comparing rank, algorithm, candidate_id, score, and mutuals.
    """
    results = recommend_all_heuristics(graph, user_id, top_k)
    rows = []
    for algo_name, recs in results.items():
        for item in recs:
            rows.append({
                "Algorithm": algo_name,
                "Rank": item["rank"],
                "Candidate ID": item["candidate_id"],
                "Score": item["score"],
                "Mutual Friends Count": item["num_mutual_friends"],
                "Mutual Friends": str(item["mutual_friends"]),
                "Candidate Degree": item["candidate_degree"],
            })
    return pd.DataFrame(rows)
