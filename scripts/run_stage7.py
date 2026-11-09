"""Stage 7 Execution Script: Real-World Dataset Integration (Stanford SNAP Facebook).
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import networkx as nx
import pandas as pd

from src.data_loader import (
    load_karate_club_graph,
    download_snap_facebook_dataset,
    load_snap_facebook_graph,
)
from src.graph_builder import validate_graph
from src.graph_analysis import compute_network_statistics
from src.embeddings import train_node2vec, extract_all_embeddings, save_embeddings
from src.link_prediction import (
    split_edges,
    sample_negative_edges,
    build_link_prediction_dataset,
    train_link_prediction_model,
    save_link_prediction_model,
)
from src.evaluation import evaluate_all_models, plot_evaluation_comparison


def main():
    print("=" * 80)
    print("STAGE 7: Real-world Dataset Integration (Stanford SNAP Facebook)")
    print("=" * 80)
    
    # 1. Download & Parse Dataset
    print("\n[Step 1] Ingesting SNAP Facebook Social Circles dataset...")
    t0 = time.time()
    txt_path = download_snap_facebook_dataset()
    t_download = time.time() - t0
    print(f"  -> File verified at: {txt_path} (Elapsed: {t_download:.2f}s)")
    
    # 2. Build Undirected Graphs
    print("\n[Step 2] Building and validating graphs...")
    # Full Facebook graph
    t0 = time.time()
    g_facebook_full = load_snap_facebook_graph()
    t_load_full = time.time() - t0
    
    val_full = validate_graph(g_facebook_full)
    print(f"  -> Full Facebook Graph : {val_full['num_nodes']} nodes, {val_full['num_edges']} edges (Loaded in {t_load_full:.2f}s)")
    assert val_full["is_valid"], "Full Facebook graph validation failed!"
    
    # Ego Network 0 (representative social circle for fast interactive exploration)
    g_facebook_ego0 = load_snap_facebook_graph(ego_user=0)
    val_ego0 = validate_graph(g_facebook_ego0)
    print(f"  -> Facebook Ego-0 Graph: {val_ego0['num_nodes']} nodes, {val_ego0['num_edges']} edges")
    assert val_ego0["is_valid"], "Facebook Ego-0 graph validation failed!"
    
    # 3. Graph Topological Statistics Comparison
    print("\n[Step 3] Comparing topological properties across datasets:")
    g_karate = load_karate_club_graph()
    
    stats_karate = compute_network_statistics(g_karate)
    stats_ego0 = compute_network_statistics(g_facebook_ego0)
    stats_full = {
        "num_nodes": g_facebook_full.number_of_nodes(),
        "num_edges": g_facebook_full.number_of_edges(),
        "density": round(nx.density(g_facebook_full), 4),
        "average_degree": round(2 * g_facebook_full.number_of_edges() / g_facebook_full.number_of_nodes(), 2),
        "is_connected": nx.is_connected(g_facebook_full),
        "average_clustering_coefficient": round(nx.average_clustering(g_facebook_full), 4),
        "transitivity": round(nx.transitivity(g_facebook_full), 4),
    }
    
    stats_comparison = pd.DataFrame([
        {
            "Dataset": "Zachary's Karate Club (Toy)",
            "Nodes (|V|)": stats_karate["num_nodes"],
            "Edges (|E|)": stats_karate["num_edges"],
            "Density": stats_karate["density"],
            "Avg Degree": stats_karate["average_degree"],
            "Avg Clustering": stats_karate["average_clustering_coefficient"],
            "Transitivity": stats_karate["transitivity"],
            "Connected": stats_karate["is_connected"],
        },
        {
            "Dataset": "SNAP Facebook Ego-0 (Medium)",
            "Nodes (|V|)": stats_ego0["num_nodes"],
            "Edges (|E|)": stats_ego0["num_edges"],
            "Density": stats_ego0["density"],
            "Avg Degree": stats_ego0["average_degree"],
            "Avg Clustering": stats_ego0["average_clustering_coefficient"],
            "Transitivity": stats_ego0["transitivity"],
            "Connected": stats_ego0["is_connected"],
        },
        {
            "Dataset": "SNAP Facebook Full (Large)",
            "Nodes (|V|)": stats_full["num_nodes"],
            "Edges (|E|)": stats_full["num_edges"],
            "Density": stats_full["density"],
            "Avg Degree": stats_full["average_degree"],
            "Avg Clustering": stats_full["average_clustering_coefficient"],
            "Transitivity": stats_full["transitivity"],
            "Connected": stats_full["is_connected"],
        },
    ])
    print(stats_comparison.to_string(index=False))
    
    # Save statistics comparison
    stats_csv = Path("data") / "processed" / "dataset_statistics_comparison.csv"
    stats_comparison.to_csv(stats_csv, index=False)
    print(f"\n  -> Saved dataset comparison to: {stats_csv}")
    
    # 4. Benchmark Link Prediction on Facebook Ego-0
    print("\n[Step 4] Running link prediction evaluation on SNAP Facebook Ego-0...")
    seed = 42
    test_ratio = 0.2
    k = 5
    
    g_train, train_pos, test_pos = split_edges(g_facebook_ego0, test_ratio=test_ratio, seed=seed)
    train_neg, test_neg = sample_negative_edges(g_facebook_ego0, len(train_pos), len(test_pos), seed=seed)
    print(f"  -> Partitioned: {len(train_pos)} train pos, {len(test_pos)} test pos, {len(test_neg)} test neg.")
    
    # Train Node2Vec on g_train
    t0 = time.time()
    n2v = train_node2vec(g_train, dimensions=16, walk_length=10, num_walks=20, seed=seed, workers=1, quiet=True)
    embeddings = extract_all_embeddings(n2v, g_train)
    t_emb = time.time() - t0
    print(f"  -> Node2Vec trained in {t_emb:.2f}s.")
    
    # Train Logistic Regression
    X_train, y_train = build_link_prediction_dataset(train_pos, train_neg, embeddings, operator="hadamard")
    clf = train_link_prediction_model(X_train, y_train, seed=seed)
    
    # Save artifacts for Facebook Ego-0
    models_dir = Path("models")
    emb_file = models_dir / "node2vec_facebook_ego0_embeddings.pkl"
    clf_file = models_dir / "link_prediction_facebook_ego0_lr.pkl"
    save_embeddings(embeddings, emb_file)
    save_link_prediction_model(clf, {"dataset": "facebook_ego0", "operator": "hadamard"}, clf_file)
    print(f"  -> Saved Facebook models to: {emb_file} & {clf_file}")
    
    # 5. Evaluate all 5 algorithms on Facebook Ego-0
    print(f"\n[Step 5] Benchmarking all 5 algorithms on Facebook Ego-0 (K={k})...")
    df_fb_eval = evaluate_all_models(
        graph_train=g_train,
        test_pos_edges=test_pos,
        test_neg_edges=test_neg,
        embeddings=embeddings,
        model_lr=clf,
        k=k,
        operator="hadamard"
    )
    
    print("\n" + "=" * 80)
    print("SNAP FACEBOOK EGO-0 EVALUATION BENCHMARK RESULTS (K=5)")
    print("=" * 80)
    print(df_fb_eval.to_string(index=False))
    print("=" * 80)
    
    # Export Facebook evaluation results
    fb_eval_csv = Path("data") / "processed" / "stage7_facebook_evaluation_results.csv"
    df_fb_eval.to_csv(fb_eval_csv, index=False)
    print(f"\n  -> Exported benchmark table to: {fb_eval_csv}")
    
    # 6. Generate and save Facebook evaluation plot
    plot_path = Path("reports") / "figures" / "facebook_evaluation_comparison.png"
    plot_evaluation_comparison(
        df_fb_eval,
        output_path=plot_path,
        k=k,
        title="SNAP Facebook Benchmark: Link Prediction & Recommendation Performance"
    )
    print(f"  -> Saved benchmark figure to: {plot_path}")
    
    print("\n" + "=" * 80)
    print("STAGE 7 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

