"""Stage 1 Execution Script: Dataset Acquisition, Graph Construction, and Analysis.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from src.data_loader import load_karate_club_graph, export_graph_to_edge_list_csv
from src.graph_builder import build_undirected_graph, validate_graph
from src.graph_analysis import compute_network_statistics, visualize_network


def main():
    print("=" * 60)
    print("STAGE 1: Dataset Acquisition and Graph Construction")
    print("=" * 60)
    
    # 1. Load Karate Club Graph
    print("\n[Step 1] Loading Zachary's Karate Club graph from NetworkX...")
    raw_graph = load_karate_club_graph()
    
    # Export raw edge list for persistence and transparency
    raw_csv_path = Path("data") / "raw" / "karate_club_edges.csv"
    export_graph_to_edge_list_csv(raw_graph, raw_csv_path)
    print(f"  -> Exported raw edge list to: {raw_csv_path}")
    
    # 2. Re-construct graph via graph_builder to validate custom pipeline
    print("\n[Step 2] Building and validating undirected graph via src.graph_builder...")
    edges = list(raw_graph.edges())
    node_attrs = {node: data for node, data in raw_graph.nodes(data=True)}
    graph = build_undirected_graph(edges, node_attributes=node_attrs)
    
    validation = validate_graph(graph)
    print(f"  -> Validation result: {validation}")
    assert validation["is_valid"], "Constructed graph failed validation!"
    assert validation["num_nodes"] == 34, f"Expected 34 nodes, got {validation['num_nodes']}"
    assert validation["num_edges"] == 78, f"Expected 78 edges, got {validation['num_edges']}"
    
    # 3. Compute Network Statistics
    print("\n[Step 3] Computing network statistics via src.graph_analysis...")
    stats = compute_network_statistics(graph)
    print("\n--- Network Statistics Summary ---")
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")
        
    # Save statistics JSON into data/processed
    stats_path = Path("data") / "processed" / "karate_club_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  -> Saved statistics to: {stats_path}")
    
    # 4. Generate Visualization
    fig_path = Path("reports") / "figures" / "karate_club_network.png"
    print(f"\n[Step 4] Visualizing network to {fig_path}...")
    saved_img = visualize_network(
        graph,
        output_path=fig_path,
        title="Zachary's Karate Club Social Network (|V|=34, |E|=78)"
    )
    print(f"  -> Visualization successfully saved to: {saved_img}")
    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
