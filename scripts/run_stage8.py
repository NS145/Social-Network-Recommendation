"""Stage 8 Execution Script: Graph Neural Networks (GCN & GraphSAGE) Link Prediction.
Compares Message-Passing GNNs against Shallow Embeddings (Node2Vec) and Topological Heuristics.
"""

import sys
from pathlib import Path
import shutil

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data_loader import load_karate_club_graph
from src.embeddings import train_node2vec, extract_all_embeddings, recommend_node2vec_cosine
from src.recommendation import recommend_all_heuristics
from src.link_prediction import (
    split_edges,
    sample_negative_edges,
    build_link_prediction_dataset,
    train_link_prediction_model,
    recommend_supervised_link_prediction,
)
from src.evaluation import evaluate_all_models
from src.gnn import (
    prepare_pyg_link_data,
    GCNLinkPredictor,
    GraphSAGELinkPredictor,
    train_gnn_model,
    evaluate_gnn_model,
    recommend_gnn_links,
)


def main():
    print("=" * 80)
    print("STAGE 8: Graph Neural Networks (GCN & GraphSAGE) Link Prediction")
    print("=" * 80)

    seed = 42
    test_ratio = 0.2
    target_user = 0
    top_k = 5
    epochs = 150
    lr = 0.01

    # 1. Load Original Graph and Split Edges
    g_orig = load_karate_club_graph()
    total_edges = g_orig.number_of_edges()
    print(f"\n[Step 1] Loaded Graph: {g_orig.number_of_nodes()} nodes, {total_edges} edges.")

    g_train, train_pos, test_pos = split_edges(
        g_orig, test_ratio=test_ratio, seed=seed, ensure_connected=True
    )
    train_neg, test_neg = sample_negative_edges(
        g_orig, num_train_neg=len(train_pos), num_test_neg=len(test_pos), seed=seed
    )
    print(f"  -> Train positive edges : {len(train_pos)}")
    print(f"  -> Test positive edges  : {len(test_pos)}")
    print(f"  -> Train negative edges : {len(train_neg)}")
    print(f"  -> Test negative edges  : {len(test_neg)}")

    # 2. Train Shallow Baseline: Node2Vec & Logistic Regression
    print("\n[Step 2] Training Shallow Baselines (Node2Vec + Heuristics + Logistic Regression)...")
    n2v = train_node2vec(
        g_train, dimensions=16, walk_length=10, num_walks=30, seed=seed, workers=1, quiet=True
    )
    embeddings = extract_all_embeddings(n2v, g_train)
    X_train, y_train = build_link_prediction_dataset(train_pos, train_neg, embeddings, operator="hadamard")
    clf = train_link_prediction_model(X_train, y_train, seed=seed)

    baseline_df = evaluate_all_models(
        graph_train=g_train,
        test_pos_edges=test_pos,
        test_neg_edges=test_neg,
        embeddings=embeddings,
        model_lr=clf,
        k=top_k,
        operator="hadamard"
    )

    # 3. Convert Data to PyTorch Geometric Format
    print("\n[Step 3] Preparing PyTorch Geometric graph data (transductive one-hot identity)...")
    pyg_data, node_to_idx, idx_to_node = prepare_pyg_link_data(g_train, feature_type="identity")
    in_channels = pyg_data.num_nodes
    hidden_channels = 32
    out_channels = 16

    print(f"  -> PyG Data: {pyg_data.num_nodes} nodes, {pyg_data.edge_index.size(1)} directed edges")
    print(f"  -> Feature dimension: {in_channels}, Hidden: {hidden_channels}, Embedding: {out_channels}")

    # 4. Train GCN Link Predictor
    print(f"\n[Step 4] Training GCN Link Predictor ({epochs} epochs, lr={lr})...")
    gcn_model = GCNLinkPredictor(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=0.1
    )
    gcn_losses = train_gnn_model(
        gcn_model,
        pyg_data,
        node_to_idx,
        train_pos,
        train_neg,
        epochs=epochs,
        lr=lr,
        seed=seed
    )
    gcn_eval = evaluate_gnn_model(gcn_model, pyg_data, node_to_idx, test_pos, test_neg)
    print(f"  -> GCN Final Train Loss: {gcn_losses[-1]:.4f}")
    print(f"  -> GCN Test ROC-AUC    : {gcn_eval['roc_auc']:.4f}")
    print(f"  -> GCN Test Avg Prec   : {gcn_eval['average_precision']:.4f}")

    # 5. Train GraphSAGE Link Predictor
    print(f"\n[Step 5] Training GraphSAGE Link Predictor ({epochs} epochs, lr={lr})...")
    sage_model = GraphSAGELinkPredictor(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=0.1
    )
    sage_losses = train_gnn_model(
        sage_model,
        pyg_data,
        node_to_idx,
        train_pos,
        train_neg,
        epochs=epochs,
        lr=lr,
        seed=seed
    )
    sage_eval = evaluate_gnn_model(sage_model, pyg_data, node_to_idx, test_pos, test_neg)
    print(f"  -> GraphSAGE Final Train Loss: {sage_losses[-1]:.4f}")
    print(f"  -> GraphSAGE Test ROC-AUC    : {sage_eval['roc_auc']:.4f}")
    print(f"  -> GraphSAGE Test Avg Prec   : {sage_eval['average_precision']:.4f}")

    # 6. Save PyTorch Models
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    gcn_path = models_dir / "gcn_link_predictor.pt"
    sage_path = models_dir / "graphsage_link_predictor.pt"
    torch.save(gcn_model.state_dict(), gcn_path)
    torch.save(sage_model.state_dict(), sage_path)
    print(f"\n[Step 6] Saved PyG models:")
    print(f"  -> GCN       : {gcn_path}")
    print(f"  -> GraphSAGE : {sage_path}")

    # 7. Build Unified Evaluation Comparison Table
    print("\n[Step 7] Compiling Global Evaluation Benchmark...")
    all_eval_rows = []
    for _, row in baseline_df.iterrows():
        all_eval_rows.append({
            "Algorithm": row["Algorithm"],
            "Category": "Topological Heuristic" if "Neighbors" in row["Algorithm"] or "Jaccard" in row["Algorithm"] or "Preferential" in row["Algorithm"] else "Shallow Representation",
            "ROC-AUC": row["ROC-AUC"],
            "Average Precision": row["Average Precision"],
        })

    all_eval_rows.append({
        "Algorithm": "GCN Link Predictor",
        "Category": "Graph Neural Network",
        "ROC-AUC": gcn_eval["roc_auc"],
        "Average Precision": gcn_eval["average_precision"],
    })
    all_eval_rows.append({
        "Algorithm": "GraphSAGE Link Predictor",
        "Category": "Graph Neural Network",
        "ROC-AUC": sage_eval["roc_auc"],
        "Average Precision": sage_eval["average_precision"],
    })

    df_gnn_comparison = pd.DataFrame(all_eval_rows)
    print("\n" + "=" * 80)
    print("GLOBAL MODEL BENCHMARK (HEURISTICS vs SHALLOW EMBEDDINGS vs GNNs)")
    print("=" * 80)
    print(df_gnn_comparison.to_string(index=False))
    print("=" * 80)

    csv_out = Path("data") / "processed" / "stage8_gnn_evaluation_results.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_gnn_comparison.to_csv(csv_out, index=False)
    print(f"  -> Exported benchmark results to: {csv_out}")

    # 8. Generate Top-K Recommendations for Target User (Node 0)
    print(f"\n[Step 8] Generating Top-{top_k} Recommendations for User {target_user}:")
    gcn_recs = recommend_gnn_links(g_orig, gcn_model, pyg_data, node_to_idx, target_user, top_k=top_k)
    sage_recs = recommend_gnn_links(g_orig, sage_model, pyg_data, node_to_idx, target_user, top_k=top_k)
    n2v_recs = recommend_node2vec_cosine(g_orig, embeddings, target_user, top_k=top_k)
    sup_recs = recommend_supervised_link_prediction(g_orig, clf, embeddings, target_user, top_k=top_k, operator="hadamard")
    heuristics = recommend_all_heuristics(g_orig, target_user, top_k=top_k)

    user_compare_rows = []
    for rank in range(1, top_k + 1):
        cn_item = heuristics["common_neighbors"][rank - 1]
        n2v_item = n2v_recs[rank - 1]
        sup_item = sup_recs[rank - 1]
        gcn_item = gcn_recs[rank - 1]
        sage_item = sage_recs[rank - 1]

        user_compare_rows.append({
            "Rank": rank,
            "Common Neighbors": f"User {cn_item['candidate_id']} ({cn_item['score']})",
            "Node2Vec Cosine": f"User {n2v_item['candidate_id']} ({n2v_item['score']:.3f})",
            "Supervised LR": f"User {sup_item['candidate_id']} ({sup_item['score']:.3f})",
            "GCN Predictor": f"User {gcn_item['candidate_id']} ({gcn_item['score']:.3f})",
            "GraphSAGE Predictor": f"User {sage_item['candidate_id']} ({sage_item['score']:.3f})",
        })

    df_user_compare = pd.DataFrame(user_compare_rows)
    print(df_user_compare.to_string(index=False))

    user_csv_out = Path("data") / "processed" / f"stage8_user_{target_user}_gnn_recommendations.csv"
    df_user_compare.to_csv(user_csv_out, index=False)
    print(f"  -> Exported User {target_user} recommendations to: {user_csv_out}")

    # 9. Publication-Quality Comparative Plotting
    print("\n[Step 9] Plotting GNN vs Shallow Baseline Comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Bar chart of ROC-AUC and Average Precision
    x_indices = np.arange(len(df_gnn_comparison))
    bar_width = 0.35

    rects1 = ax1.bar(
        x_indices - bar_width / 2,
        df_gnn_comparison["ROC-AUC"],
        bar_width,
        label="ROC-AUC",
        color="#1f77b4",
        edgecolor="black"
    )
    rects2 = ax1.bar(
        x_indices + bar_width / 2,
        df_gnn_comparison["Average Precision"],
        bar_width,
        label="Average Precision",
        color="#2ca02c",
        edgecolor="black"
    )

    ax1.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax1.set_title("Link Prediction Performance on Held-Out Test Set", fontsize=12, fontweight="bold")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(df_gnn_comparison["Algorithm"], rotation=30, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Add score annotations above bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    # Panel 2: GNN Training Loss Convergence
    ax2.plot(range(1, epochs + 1), gcn_losses, label="GCN (Binary Cross-Entropy)", color="#ff7f0e", lw=2)
    ax2.plot(range(1, epochs + 1), sage_losses, label="GraphSAGE (Binary Cross-Entropy)", color="#9467bd", lw=2)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("BCE Loss", fontsize=11, fontweight="bold")
    ax2.set_title("GNN Training Loss Convergence (Message Passing)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = Path("reports") / "figures" / "gnn_vs_shallow_embeddings.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  -> Saved figure to: {fig_path}")

    # Copy to artifact dir if artifact directory exists
    artifact_dir = Path(r"C:\Users\N S Balaji\.gemini\antigravity\brain\c4a81176-909b-4a81-b073-291c5ca1f783")
    if artifact_dir.exists():
        dest = artifact_dir / "gnn_vs_shallow_embeddings.png"
        shutil.copyfile(fig_path, dest)
        print(f"  -> Copied figure to artifact dir: {dest}")

    print("\n" + "=" * 80)
    print("STAGE 8 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
