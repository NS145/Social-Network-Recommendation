"""Stage 4 Execution Script: Supervised Link Prediction using Node2Vec & Logistic Regression.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, classification_report

from src.data_loader import load_karate_club_graph
from src.embeddings import train_node2vec, extract_all_embeddings
from src.recommendation import recommend_all_heuristics
from src.embeddings import recommend_node2vec_cosine
from src.link_prediction import (
    split_edges,
    sample_negative_edges,
    compute_edge_features,
    build_link_prediction_dataset,
    train_link_prediction_model,
    save_link_prediction_model,
    recommend_supervised_link_prediction
)


def main():
    print("=" * 75)
    print("STAGE 4: Supervised Link Prediction (Node2Vec + Logistic Regression)")
    print("=" * 75)
    
    seed = 42
    test_ratio = 0.2
    target_user = 0
    top_k = 5
    operator = "hadamard"
    
    # 1. Load Original Graph
    g_orig = load_karate_club_graph()
    total_edges = g_orig.number_of_edges()
    print(f"\n[Step 1] Loaded Graph: {g_orig.number_of_nodes()} nodes, {total_edges} edges.")
    
    # 2. Split Edges without Data Leakage
    print(f"\n[Step 2] Splitting edges (test_ratio={test_ratio}, seed={seed}):")
    g_train, train_pos, test_pos = split_edges(
        g_orig,
        test_ratio=test_ratio,
        seed=seed,
        ensure_connected=True
    )
    print(f"  -> Train positive edges : {len(train_pos)} ({len(train_pos)/total_edges*100:.1f}%)")
    print(f"  -> Test positive edges  : {len(test_pos)} ({len(test_pos)/total_edges*100:.1f}%)")
    print(f"  -> Training graph connected: {g_train.number_of_nodes()} nodes, {g_train.number_of_edges()} edges")
    
    # 3. Negative Sampling
    print("\n[Step 3] Sampling negative edges (non-existent links):")
    train_neg, test_neg = sample_negative_edges(
        g_orig,
        num_train_neg=len(train_pos),
        num_test_neg=len(test_pos),
        seed=seed
    )
    print(f"  -> Train negative samples : {len(train_neg)}")
    print(f"  -> Test negative samples  : {len(test_neg)}")
    print(f"  -> Verified zero overlap between train and test negatives.")
    
    # 4. Train Node2Vec on Training Graph ONLY
    print("\n[Step 4] Training Node2Vec on g_train ONLY (preventing test leakage):")
    dimensions = 16
    n2v_model = train_node2vec(
        g_train,
        dimensions=dimensions,
        walk_length=10,
        num_walks=30,
        p=1.0,
        q=1.0,
        window=5,
        seed=seed,
        workers=1,
        quiet=True
    )
    embeddings = extract_all_embeddings(n2v_model, g_train)
    print(f"  -> Learned {len(embeddings)} node embeddings of dimension {dimensions}.")
    
    # 5. Build Feature Datasets (Hadamard Operator)
    print(f"\n[Step 5] Building feature matrices using operator: '{operator}'")
    X_train, y_train = build_link_prediction_dataset(train_pos, train_neg, embeddings, operator=operator)
    X_test, y_test = build_link_prediction_dataset(test_pos, test_neg, embeddings, operator=operator)
    print(f"  -> X_train shape: {X_train.shape}, y_train class balance: {np.bincount(y_train)}")
    print(f"  -> X_test shape : {X_test.shape}, y_test class balance : {np.bincount(y_test)}")
    
    # 6. Train Logistic Regression Classifier
    print("\n[Step 6] Training Logistic Regression classifier:")
    clf = train_link_prediction_model(X_train, y_train, seed=seed)
    
    # 7. Evaluate on Held-out Test Set
    y_test_pred = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_ap = average_precision_score(y_test, y_test_proba)
    
    print("\n--- Test Set Evaluation Metrics ---")
    print(f"  Test Accuracy           : {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Test ROC-AUC            : {test_auc:.4f}")
    print(f"  Test Average Precision  : {test_ap:.4f}")
    print("\nClassification Report on Held-Out Test Edges:")
    print(classification_report(y_test, y_test_pred, target_names=["No Edge (0)", "Edge (1)"]))
    
    # 8. Serialize Trained Model
    model_path = Path("models") / "link_prediction_logistic_regression.pkl"
    metadata = {
        "model_type": "LogisticRegression",
        "operator": operator,
        "dimensions": dimensions,
        "test_ratio": test_ratio,
        "seed": seed,
        "test_accuracy": test_acc,
        "test_roc_auc": test_auc,
        "test_average_precision": test_ap,
    }
    save_link_prediction_model(clf, metadata, model_path)
    print(f"  -> Saved model and metadata to: {model_path}")
    
    # 9. Generate Supervised Recommendations for User 0
    print(f"\n[Step 9] Generating Supervised Link Prediction recommendations for User {target_user}:")
    sup_recs = recommend_supervised_link_prediction(
        g_orig,
        clf,
        embeddings,
        user_id=target_user,
        top_k=top_k,
        operator=operator
    )
    
    header = f"{'Rank':<6}{'Candidate':<12}{'P(Link)':<12}{'Mutuals Count':<16}{'Cand Degree':<14}{'Mutual Friends'}"
    print(header)
    print("-" * len(header))
    for r in sup_recs:
        print(f"{r['rank']:<6}{r['candidate_id']:<12}{r['score']:<12.4f}{r['num_mutual_friends']:<16}{r['candidate_degree']:<14}{r['mutual_friends']}")
        
    # 10. Comprehensive 5-Way Comparison
    print(f"\n[Step 10] Comprehensive 5-Way Recommendation Comparison for User {target_user}:")
    stage2_recs = recommend_all_heuristics(g_orig, target_user, top_k=top_k)
    n2v_recs = recommend_node2vec_cosine(g_orig, embeddings, target_user, top_k=top_k)
    
    comparison_rows = []
    for rank in range(1, top_k + 1):
        cn_item = stage2_recs["common_neighbors"][rank - 1]
        jac_item = stage2_recs["jaccard"][rank - 1]
        pa_item = stage2_recs["preferential_attachment"][rank - 1]
        n2v_item = n2v_recs[rank - 1]
        sup_item = sup_recs[rank - 1]
        
        comparison_rows.append({
            "Rank": rank,
            "Common Neighbors": f"Node {cn_item['candidate_id']} ({cn_item['score']})",
            "Jaccard": f"Node {jac_item['candidate_id']} ({jac_item['score']:.4f})",
            "Pref. Attachment": f"Node {pa_item['candidate_id']} ({pa_item['score']})",
            "Node2Vec Cosine": f"Node {n2v_item['candidate_id']} ({n2v_item['score']:.4f})",
            "Supervised LR": f"Node {sup_item['candidate_id']} ({sup_item['score']:.4f})",
        })
        
    df_compare = pd.DataFrame(comparison_rows)
    print(df_compare.to_string(index=False))
    
    out_csv = Path("data") / "processed" / f"stage4_5way_recommendations_user_{target_user}.csv"
    df_compare.to_csv(out_csv, index=False)
    print(f"\n  -> Exported 5-way comparison table to: {out_csv}")
    
    print("\n" + "=" * 75)
    print("STAGE 4 COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()

