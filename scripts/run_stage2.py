"""Stage 2 Execution Script: Traditional Graph-based Recommendation Algorithms.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
from src.data_loader import load_karate_club_graph
from src.recommendation import (
    recommend_all_heuristics,
    compare_recommendations_dataframe,
    get_candidate_users
)


def display_recommendations_table(algo_name: str, recs: list):
    print(f"\n--- {algo_name.upper()} Top {len(recs)} Recommendations ---")
    header = f"{'Rank':<6}{'Candidate':<12}{'Score':<10}{'Mutuals Count':<16}{'Cand Degree':<14}{'Mutual Friends'}"
    print(header)
    print("-" * len(header))
    for r in recs:
        mutuals_str = str(r['mutual_friends'])
        print(f"{r['rank']:<6}{r['candidate_id']:<12}{r['score']:<10}{r['num_mutual_friends']:<16}{r['candidate_degree']:<14}{mutuals_str}")


def main():
    print("=" * 70)
    print("STAGE 2: Traditional Graph-based Recommendation Algorithms")
    print("=" * 70)
    
    graph = load_karate_club_graph()
    target_user = 0
    top_k = 5
    
    existing_friends = sorted(list(graph.neighbors(target_user)))
    candidates = sorted(list(get_candidate_users(graph, target_user)))
    
    print(f"\nSelected User ID : {target_user} (Affiliation: {graph.nodes[target_user].get('club', 'N/A')})")
    print(f"User Degree      : {graph.degree(target_user)}")
    print(f"Existing Friends : {existing_friends} (Count: {len(existing_friends)})")
    print(f"Candidate Users  : {candidates} (Count: {len(candidates)})")
    
    # 1. Run All Heuristics
    results = recommend_all_heuristics(graph, target_user, top_k=top_k)
    
    display_recommendations_table("Common Neighbors", results["common_neighbors"])
    display_recommendations_table("Jaccard Similarity", results["jaccard"])
    display_recommendations_table("Preferential Attachment", results["preferential_attachment"])
    
    # 2. Export Comparison DataFrame
    df_comparison = compare_recommendations_dataframe(graph, target_user, top_k=top_k)
    output_csv = Path("data") / "processed" / f"stage2_recommendations_user_{target_user}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_comparison.to_csv(output_csv, index=False)
    print(f"\n-> Exported consolidated recommendation comparison to: {output_csv}")
    
    # 3. Test edge case: peripheral / isolated node handling
    print("\n[Edge Case Verification] Testing a low-degree peripheral user (Node 24):")
    p_recs = recommend_all_heuristics(graph, user_id=24, top_k=3)
    display_recommendations_table("Common Neighbors (Node 24)", p_recs["common_neighbors"])
    
    print("\n" + "=" * 70)
    print("STAGE 2 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

