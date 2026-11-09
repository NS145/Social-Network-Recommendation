"""Stage 5 Execution Script: Rigorous Model Evaluation & Algorithmic Comparison.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
from src.data_loader import load_karate_club_graph
from src.embeddings import train_node2vec, extract_all_embeddings
from src.link_prediction import (
    split_edges,
    sample_negative_edges,
    build_link_prediction_dataset,
    train_link_prediction_model,
)
from src.evaluation import evaluate_all_models, plot_evaluation_comparison


def main():
    print("=" * 80)
    print("STAGE 5: Comprehensive Model Evaluation and Algorithmic Comparison")
    print("=" * 80)
    
    seed = 42
    test_ratio = 0.2
    k = 5
    operator = "hadamard"
    
    # 1. Load Graph and Split
    g_orig = load_karate_club_graph()
    print(f"\n[Step 1] Loaded Graph: {g_orig.number_of_nodes()} nodes, {g_orig.number_of_edges()} edges.")
    
    g_train, train_pos, test_pos = split_edges(g_orig, test_ratio=test_ratio, seed=seed)
    train_neg, test_neg = sample_negative_edges(g_orig, len(train_pos), len(test_pos), seed=seed)
    print(f"  -> Partitioned: {len(train_pos)} train pos, {len(test_pos)} test pos, {len(test_neg)} test neg.")
    
    # 2. Train Embeddings on G_train ONLY
    print("\n[Step 2] Training Node2Vec on g_train ONLY (preventing test leakage)...")
    n2v = train_node2vec(g_train, dimensions=16, walk_length=10, num_walks=30, seed=seed, workers=1, quiet=True)
    embeddings = extract_all_embeddings(n2v, g_train)
    
    # 3. Train Supervised Classifier
    print("\n[Step 3] Training Logistic Regression on training edge embeddings...")
    X_train, y_train = build_link_prediction_dataset(train_pos, train_neg, embeddings, operator=operator)
    clf = train_link_prediction_model(X_train, y_train, seed=seed)
    
    # 4. Run Unified Evaluation Across All 5 Algorithms
    print(f"\n[Step 4] Running unified benchmark across all 5 algorithms (Top-K={k})...")
    df_results = evaluate_all_models(
        graph_train=g_train,
        test_pos_edges=test_pos,
        test_neg_edges=test_neg,
        embeddings=embeddings,
        model_lr=clf,
        k=k,
        operator=operator
    )
    
    print("\n" + "=" * 80)
    print(f"MODEL EVALUATION BENCHMARK RESULTS (K={k})")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)
    
    # 5. Save Results to data/processed/
    csv_path = Path("data") / "processed" / "model_evaluation_results.csv"
    json_path = Path("data") / "processed" / "model_evaluation_results.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(df_results.to_dict(orient="records"), f, indent=2)
    print(f"\n[Step 5] Saved evaluation tables:")
    print(f"  -> CSV  : {csv_path}")
    print(f"  -> JSON : {json_path}")
    
    # 6. Generate and Save Comparison Plots
    plot_path = Path("reports") / "figures" / "model_evaluation_comparison.png"
    plot_evaluation_comparison(df_results, output_path=plot_path, k=k)
    print(f"\n[Step 6] Saved benchmark visualization:")
    print(f"  -> Figure : {plot_path}")
    
    print("\n" + "=" * 80)
    print("STAGE 5 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

