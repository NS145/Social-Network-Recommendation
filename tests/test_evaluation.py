"""Unit tests for Stage 5: Model Evaluation and Algorithmic Benchmarking.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import networkx as nx

from src.data_loader import load_karate_club_graph
from src.embeddings import train_node2vec, extract_all_embeddings
from src.link_prediction import (
    split_edges,
    sample_negative_edges,
    build_link_prediction_dataset,
    train_link_prediction_model,
)
from src.evaluation import (
    evaluate_edge_classification,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_all_models,
    plot_evaluation_comparison,
)


# ============================================================================
# Test Classification Metrics
# ============================================================================

def test_evaluate_edge_classification_perfect():
    """Verify metrics for perfect classification scores."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1])
    metrics = evaluate_edge_classification(y_true, y_score)
    
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0


def test_evaluate_edge_classification_random():
    """Verify metrics for inverse classification scores."""
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_edge_classification(y_true, y_score)
    
    assert metrics["roc_auc"] == 0.0


# ============================================================================
# Test Ranking Metrics
# ============================================================================

def test_compute_precision_at_k():
    """Verify Precision@K calculation."""
    recommended = [1, 2, 3, 4, 5]
    ground_truth = {2, 4, 10}
    
    # 2 hits out of 5 -> 0.4
    prec = compute_precision_at_k(recommended, ground_truth, k=5)
    assert prec == 0.4
    
    # 1 hit out of 2 -> 0.5
    prec2 = compute_precision_at_k(recommended, ground_truth, k=2)
    assert prec2 == 0.5
    
    with pytest.raises(ValueError, match="positive integer"):
        compute_precision_at_k(recommended, ground_truth, k=0)


def test_compute_recall_at_k():
    """Verify Recall@K calculation."""
    recommended = [1, 2, 3, 4, 5]
    ground_truth = {2, 4, 10}
    
    # 2 hits out of 3 relevant items -> 2/3
    rec = compute_recall_at_k(recommended, ground_truth, k=5)
    assert abs(rec - (2 / 3)) < 1e-4
    
    # Empty ground truth edge case
    assert compute_recall_at_k(recommended, set(), k=5) == 0.0


# ============================================================================
# Test Integrated Model Benchmark
# ============================================================================

def test_evaluate_all_models_dataframe_output():
    """Verify end-to-end evaluation pipeline output structure and values."""
    g = load_karate_club_graph()
    g_train, train_pos, test_pos = split_edges(g, test_ratio=0.2, seed=42)
    train_neg, test_neg = sample_negative_edges(g, len(train_pos), len(test_pos), seed=42)
    
    n2v = train_node2vec(g_train, dimensions=16, walk_length=6, num_walks=10, seed=42, workers=1, quiet=True)
    embeddings = extract_all_embeddings(n2v, g_train)
    
    X_train, y_train = build_link_prediction_dataset(train_pos, train_neg, embeddings)
    clf = train_link_prediction_model(X_train, y_train, seed=42)
    
    df_eval = evaluate_all_models(
        graph_train=g_train,
        test_pos_edges=test_pos,
        test_neg_edges=test_neg,
        embeddings=embeddings,
        model_lr=clf,
        k=5
    )
    
    assert isinstance(df_eval, pd.DataFrame)
    assert len(df_eval) == 5
    expected_cols = {"Algorithm", "Precision@5", "Recall@5", "ROC-AUC", "Average Precision"}
    assert expected_cols.issubset(set(df_eval.columns))
    
    # Check algorithms presence
    expected_algos = {
        "Common Neighbors",
        "Jaccard Similarity",
        "Preferential Attachment",
        "Node2Vec Cosine",
        "Node2Vec + Logistic Regression"
    }
    assert set(df_eval["Algorithm"].unique()) == expected_algos
    
    # Check that all metrics are valid bounded floats
    for col in ["Precision@5", "Recall@5", "ROC-AUC", "Average Precision"]:
        vals = df_eval[col].values
        assert all(0.0 <= v <= 1.0 for v in vals)


def test_plot_evaluation_comparison(tmp_path):
    """Verify evaluation plotting generates valid PNG image."""
    mock_df = pd.DataFrame([
        {"Algorithm": "Algo A", "Precision@5": 0.2, "Recall@5": 0.4, "ROC-AUC": 0.75, "Average Precision": 0.70},
        {"Algorithm": "Algo B", "Precision@5": 0.3, "Recall@5": 0.5, "ROC-AUC": 0.85, "Average Precision": 0.80},
    ])
    fig_file = tmp_path / "test_eval_plot.png"
    result = plot_evaluation_comparison(mock_df, output_path=fig_file, k=5)
    
    assert result.is_file()
    assert result.stat().st_size > 1000
