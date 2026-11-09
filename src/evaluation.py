"""Model evaluation and benchmarking module.
Provides rigorous evaluation across traditional heuristics, embedding similarity,
and supervised link prediction models.
Computes:
- Global link classification: ROC-AUC, Average Precision (PR-AUC)
- Top-K recommendation ranking: Precision@K, Recall@K
- Comparative performance tables and publication-ready plots.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

from src.embeddings import compute_cosine_similarity
from src.link_prediction import compute_edge_features, build_link_prediction_dataset


def evaluate_edge_classification(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Dict[str, float]:
    """Calculate ROC-AUC and Average Precision on binary link prediction scores.
    
    Args:
        y_true: Ground-truth binary labels (1 = edge, 0 = non-edge).
        y_score: Continuous prediction or heuristic score.
        
    Returns:
        Dict with 'roc_auc' and 'average_precision'.
    """
    roc_auc = float(roc_auc_score(y_true, y_score))
    avg_precision = float(average_precision_score(y_true, y_score))
    return {
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(avg_precision, 4)
    }


def compute_precision_at_k(
    recommended: List[Any],
    ground_truth: Set[Any],
    k: int = 5
) -> float:
    """Calculate Precision@K: fraction of top-K recommendations that are true positive links.
    
    Formula:
        Precision@K = |TopK ∩ GroundTruth| / K
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    top_k = recommended[:k]
    hits = len(set(top_k).intersection(ground_truth))
    return hits / k


def compute_recall_at_k(
    recommended: List[Any],
    ground_truth: Set[Any],
    k: int = 5
) -> float:
    """Calculate Recall@K: fraction of ground-truth test links retrieved in top-K.
    
    Formula:
        Recall@K = |TopK ∩ GroundTruth| / |GroundTruth|
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if len(ground_truth) == 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k).intersection(ground_truth))
    return hits / len(ground_truth)


def evaluate_all_models(
    graph_train: nx.Graph,
    test_pos_edges: List[Tuple[Any, Any]],
    test_neg_edges: List[Tuple[Any, Any]],
    embeddings: Dict[Any, np.ndarray],
    model_lr: LogisticRegression,
    k: int = 5,
    operator: str = "hadamard"
) -> pd.DataFrame:
    """Run an identical, fair benchmark across all 5 recommendation methods.
    
    Models Evaluated:
    1. Common Neighbors
    2. Jaccard Similarity
    3. Preferential Attachment
    4. Node2Vec Cosine Similarity
    5. Node2Vec + Logistic Regression (Supervised)
    
    Evaluated under identical test conditions:
    - Classification metrics on identical test_pos and test_neg pairs.
    - Ranking metrics (Precision@K, Recall@K) on identical candidate universes per test user.
    
    Args:
        graph_train: Training graph G_train (edges isolated from test set).
        test_pos_edges: Ground-truth positive test edges.
        test_neg_edges: Disjoint negative test edges.
        embeddings: Node2Vec embeddings trained on graph_train.
        model_lr: Trained LogisticRegression classifier.
        k: Cutoff rank for Precision@K and Recall@K.
        operator: Edge feature operator used for logistic regression.
        
    Returns:
        pd.DataFrame: Comparison table containing all metrics.
    """
    test_pairs = test_pos_edges + test_neg_edges
    y_true = np.array([1] * len(test_pos_edges) + [0] * len(test_neg_edges), dtype=np.int32)
    
    # -------------------------------------------------------------
    # 1. Classification Scores on Held-Out Test Pairs
    # -------------------------------------------------------------
    # Common Neighbors
    cn_scores = [len(list(nx.common_neighbors(graph_train, u, v))) for u, v in test_pairs]
    
    # Jaccard
    jac_map = {(u, v): p for u, v, p in nx.jaccard_coefficient(graph_train, test_pairs)}
    jac_scores = [jac_map.get((u, v), jac_map.get((v, u), 0.0)) for u, v in test_pairs]
    
    # Preferential Attachment
    pa_map = {(u, v): p for u, v, p in nx.preferential_attachment(graph_train, test_pairs)}
    pa_scores = [pa_map.get((u, v), pa_map.get((v, u), 0.0)) for u, v in test_pairs]
    
    # Node2Vec Cosine
    n2v_scores = [compute_cosine_similarity(embeddings[u], embeddings[v]) for u, v in test_pairs]
    
    # Supervised Logistic Regression
    X_test, _ = build_link_prediction_dataset(test_pos_edges, test_neg_edges, embeddings, operator=operator)
    lr_scores = model_lr.predict_proba(X_test)[:, 1]
    
    all_scores = {
        "Common Neighbors": np.array(cn_scores),
        "Jaccard Similarity": np.array(jac_scores),
        "Preferential Attachment": np.array(pa_scores),
        "Node2Vec Cosine": np.array(n2v_scores),
        "Node2Vec + Logistic Regression": lr_scores,
    }
    
    # -------------------------------------------------------------
    # 2. Ranking Evaluation: Precision@K & Recall@K per Test User
    # -------------------------------------------------------------
    user_test_targets: Dict[Any, Set[Any]] = defaultdict(set)
    for u, v in test_pos_edges:
        user_test_targets[u].add(v)
        user_test_targets[v].add(u)
        
    eval_users = list(user_test_targets.keys())
    ranking_metrics: Dict[str, Dict[str, List[float]]] = {
        name: {"p": [], "r": []} for name in all_scores.keys()
    }
    
    for u in eval_users:
        relevant_targets = user_test_targets[u]
        # Candidate set: nodes not connected to u in graph_train (excluding u)
        candidates = [c for c in graph_train.nodes() if c != u and not graph_train.has_edge(u, c)]
        if not candidates:
            continue
            
        cand_pairs = [(u, c) for c in candidates]
        
        # CN ranking
        cn_user_scores = {c: len(list(nx.common_neighbors(graph_train, u, c))) for c in candidates}
        top_cn = sorted(candidates, key=lambda c: (-cn_user_scores[c], str(c)))[:k]
        
        # Jaccard ranking
        jac_user_scores = {c: p for _, c, p in nx.jaccard_coefficient(graph_train, cand_pairs)}
        top_jac = sorted(candidates, key=lambda c: (-jac_user_scores.get(c, 0.0), str(c)))[:k]
        
        # PA ranking
        pa_user_scores = {c: p for _, c, p in nx.preferential_attachment(graph_train, cand_pairs)}
        top_pa = sorted(candidates, key=lambda c: (-pa_user_scores.get(c, 0.0), str(c)))[:k]
        
        # Node2Vec Cosine ranking
        n2v_user_scores = {c: compute_cosine_similarity(embeddings[u], embeddings[c]) for c in candidates}
        top_n2v = sorted(candidates, key=lambda c: (-n2v_user_scores[c], str(c)))[:k]
        
        # Supervised LR ranking
        X_user_cand = np.array([
            compute_edge_features(embeddings, u, c, operator=operator) for c in candidates
        ], dtype=np.float32)
        lr_probs = model_lr.predict_proba(X_user_cand)[:, 1]
        lr_user_scores = {c: p for c, p in zip(candidates, lr_probs)}
        top_lr = sorted(candidates, key=lambda c: (-lr_user_scores[c], str(c)))[:k]
        
        topk_dict = {
            "Common Neighbors": top_cn,
            "Jaccard Similarity": top_jac,
            "Preferential Attachment": top_pa,
            "Node2Vec Cosine": top_n2v,
            "Node2Vec + Logistic Regression": top_lr,
        }
        
        for name, topk_list in topk_dict.items():
            ranking_metrics[name]["p"].append(compute_precision_at_k(topk_list, relevant_targets, k=k))
            ranking_metrics[name]["r"].append(compute_recall_at_k(topk_list, relevant_targets, k=k))
            
    # -------------------------------------------------------------
    # 3. Assemble Consolidated Comparison DataFrame
    # -------------------------------------------------------------
    rows = []
    for name, scores in all_scores.items():
        clf_metrics = evaluate_edge_classification(y_true, scores)
        mean_p = float(np.mean(ranking_metrics[name]["p"])) if ranking_metrics[name]["p"] else 0.0
        mean_r = float(np.mean(ranking_metrics[name]["r"])) if ranking_metrics[name]["r"] else 0.0
        
        rows.append({
            "Algorithm": name,
            f"Precision@{k}": round(mean_p, 4),
            f"Recall@{k}": round(mean_r, 4),
            "ROC-AUC": clf_metrics["roc_auc"],
            "Average Precision": clf_metrics["average_precision"],
        })
        
    df_results = pd.DataFrame(rows)
    return df_results


def plot_evaluation_comparison(
    df_results: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    k: int = 5,
    title: str = "Benchmark: Link Prediction & Recommendation Performance"
) -> Path:
    """Generate side-by-side comparative bar charts for ranking and classification metrics.
    
    Args:
        df_results: Comparison DataFrame from evaluate_all_models.
        output_path: Path to save PNG figure.
        k: K value for plot annotations.
        title: Overall figure title.
        
    Returns:
        Path: Path to saved figure.
    """
    if output_path is None:
        output_path = Path("reports") / "figures" / "model_evaluation_comparison.png"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    algorithms = df_results["Algorithm"].tolist()
    precisions = df_results[f"Precision@{k}"].tolist()
    recalls = df_results[f"Recall@{k}"].tolist()
    aucs = df_results["ROC-AUC"].tolist()
    aps = df_results["Average Precision"].tolist()
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Subplot 1: Top-K Ranking Metrics
    ax1.bar(x - width/2, precisions, width, label=f"Precision@{k}", color="#2ca02c", alpha=0.85)
    ax1.bar(x + width/2, recalls, width, label=f"Recall@{k}", color="#1f77b4", alpha=0.85)
    ax1.set_title(f"Recommendation Ranking Quality (Top-{k})", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    
    # Subplot 2: Global Link Classification Metrics
    ax2.bar(x - width/2, aucs, width, label="ROC-AUC", color="#ff7f0e", alpha=0.85)
    ax2.bar(x + width/2, aps, width, label="Average Precision", color="#9467bd", alpha=0.85)
    ax2.set_title("Global Link Classification Performance", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    return output_path
