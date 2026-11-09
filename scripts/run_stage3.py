"""Stage 3 Execution Script: Node2Vec Representation Learning & Cosine Recommendations.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.data_loader import load_karate_club_graph
from src.recommendation import (
    recommend_all_heuristics,
    get_candidate_users
)
from src.embeddings import (
    train_node2vec,
    extract_all_embeddings,
    save_embeddings,
    save_node2vec_model,
    get_node_embedding,
    recommend_node2vec_cosine
)


def main():
    print("=" * 70)
    print("STAGE 3: Node2Vec Representation Learning & Cosine Recommendation")
    print("=" * 70)
    
    # 1. Load Graph
    graph = load_karate_club_graph()
    target_user = 0
    top_k = 5
    
    print(f"\n[Step 1] Loaded Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
    
    # 2. Configure & Train Node2Vec
    dimensions = 16
    walk_length = 10
    num_walks = 30
    p = 1.0
    q = 1.0
    window = 5
    seed = 42
    
    print("\n[Step 2] Training Node2Vec with parameters:")
    print(f"  - dimensions  : {dimensions}")
    print(f"  - walk_length : {walk_length}")
    print(f"  - num_walks   : {num_walks}")
    print(f"  - p (return)  : {p}")
    print(f"  - q (in-out)  : {q}")
    print(f"  - window      : {window}")
    print(f"  - seed        : {seed}")
    
    model = train_node2vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        window=window,
        seed=seed,
        workers=1,
        quiet=True
    )
    print("  -> Node2Vec training completed successfully.")
    
    # 3. Extract and Inspect Embeddings
    print("\n[Step 3] Extracting and inspecting learned continuous embeddings:")
    embeddings = extract_all_embeddings(model, graph)
    
    for sample_node in [0, 33, 16]:
        vec = embeddings[sample_node]
        formatted_vec = np.array2string(vec[:6], precision=3, separator=', ')
        affiliation = graph.nodes[sample_node].get('club', 'N/A')
        print(f"  - Node {sample_node:2d} ({affiliation:7s}, deg={graph.degree(sample_node):2d}) vector [{dimensions}d]: {formatted_vec[:-1]}, ...]")
        
    # 4. Serialize Model & Embeddings to models/
    models_dir = Path("models")
    emb_path = models_dir / "node2vec_karate_embeddings.pkl"
    model_path = models_dir / "node2vec_karate.model"
    save_embeddings(embeddings, emb_path)
    save_node2vec_model(model, model_path)
    print(f"\n[Step 4] Serialized artifacts saved:")
    print(f"  -> Embeddings dictionary : {emb_path}")
    print(f"  -> Gensim Word2Vec model : {model_path}")
    
    # 5. Generate Node2Vec Cosine Recommendations
    print(f"\n[Step 5] Generating Top {top_k} Node2Vec recommendations for User {target_user}:")
    n2v_recs = recommend_node2vec_cosine(graph, embeddings, target_user, top_k=top_k)
    
    header = f"{'Rank':<6}{'Candidate':<12}{'Cosine Sim':<14}{'Mutuals Count':<16}{'Cand Degree':<14}{'Mutual Friends'}"
    print(header)
    print("-" * len(header))
    for r in n2v_recs:
        print(f"{r['rank']:<6}{r['candidate_id']:<12}{r['score']:<14.4f}{r['num_mutual_friends']:<16}{r['candidate_degree']:<14}{r['mutual_friends']}")
        
    # 6. Side-by-Side Comparison with Stage 2 Heuristics
    print(f"\n[Step 6] Comprehensive Comparison Table for User {target_user}:")
    stage2_recs = recommend_all_heuristics(graph, target_user, top_k=top_k)
    
    comparison_rows = []
    for rank in range(1, top_k + 1):
        cn_item = stage2_recs["common_neighbors"][rank - 1]
        jac_item = stage2_recs["jaccard"][rank - 1]
        pa_item = stage2_recs["preferential_attachment"][rank - 1]
        n2v_item = n2v_recs[rank - 1]
        
        comparison_rows.append({
            "Rank": rank,
            "Common Neighbors": f"Node {cn_item['candidate_id']} (score={cn_item['score']})",
            "Jaccard": f"Node {jac_item['candidate_id']} (score={jac_item['score']})",
            "Pref. Attachment": f"Node {pa_item['candidate_id']} (score={pa_item['score']})",
            "Node2Vec Cosine": f"Node {n2v_item['candidate_id']} (sim={n2v_item['score']})",
        })
        
    df_compare = pd.DataFrame(comparison_rows)
    print(df_compare.to_string(index=False))
    
    # Export comparison to CSV
    csv_out = Path("data") / "processed" / f"stage3_recommendations_comparison_user_{target_user}.csv"
    df_compare.to_csv(csv_out, index=False)
    print(f"\n  -> Comparison exported to: {csv_out}")
    
    print("\n" + "=" * 70)
    print("STAGE 3 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

