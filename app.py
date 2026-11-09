"""Streamlit Application Entry Point.
Social Network Recommendation Engine using Graph Representation Learning and Link Prediction.
Supports multi-dataset selection: Zachary's Karate Club and Stanford SNAP Facebook.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from sklearn.linear_model import LogisticRegression

# Import modular src components
from src.data_loader import load_karate_club_graph, load_snap_facebook_graph
from src.graph_analysis import compute_network_statistics, create_interactive_network_figure
from src.recommendation import (
    get_candidate_users,
    recommend_all_heuristics,
    compare_recommendations_dataframe,
)
from src.embeddings import (
    train_node2vec,
    extract_all_embeddings,
    load_embeddings,
    save_embeddings,
    recommend_node2vec_cosine,
)
from src.link_prediction import (
    split_edges,
    sample_negative_edges,
    build_link_prediction_dataset,
    train_link_prediction_model,
    load_link_prediction_model,
    save_link_prediction_model,
    recommend_supervised_link_prediction,
)
from src.evaluation import evaluate_all_models


# ============================================================================
# Cached Resource Loaders
# ============================================================================

@st.cache_resource(show_spinner="Loading social network graph...")
def get_cached_graph_and_stats(dataset_key: str = "karate") -> Tuple[nx.Graph, Dict[str, Any]]:
    """Load graph and compute network statistics based on dataset selection."""
    if dataset_key == "facebook_ego0":
        graph = load_snap_facebook_graph(ego_user=0)
    elif dataset_key == "facebook_full":
        graph = load_snap_facebook_graph()
    else:
        graph = load_karate_club_graph()
        
    stats = compute_network_statistics(graph)
    return graph, stats


@st.cache_resource(show_spinner="Loading or training Machine Learning models...")
def get_cached_models_and_embeddings(
    _graph: nx.Graph,
    dataset_key: str = "karate"
) -> Tuple[Dict[Any, np.ndarray], LogisticRegression, Dict[str, Any]]:
    """Load pre-trained Node2Vec embeddings and Logistic Regression model.
    
    Gracefully handles missing files by training them on the fly if necessary.
    """
    models_dir = Path("models")
    if dataset_key == "facebook_ego0":
        emb_path = models_dir / "node2vec_facebook_ego0_embeddings.pkl"
        clf_path = models_dir / "link_prediction_facebook_ego0_lr.pkl"
    elif dataset_key == "facebook_full":
        emb_path = models_dir / "node2vec_facebook_full_embeddings.pkl"
        clf_path = models_dir / "link_prediction_facebook_full_lr.pkl"
    else:
        emb_path = models_dir / "node2vec_karate_embeddings.pkl"
        clf_path = models_dir / "link_prediction_logistic_regression.pkl"
        
    # Check if pre-trained artifacts exist
    if emb_path.is_file() and clf_path.is_file():
        try:
            embeddings = load_embeddings(emb_path)
            clf, metadata = load_link_prediction_model(clf_path)
            return embeddings, clf, metadata
        except Exception:
            pass  # Fall back to training
            
    # Train if missing
    seed = 42
    test_ratio = 0.2
    g_train, train_pos, test_pos = split_edges(_graph, test_ratio=test_ratio, seed=seed)
    train_neg, test_neg = sample_negative_edges(_graph, len(train_pos), len(test_pos), seed=seed)
    
    # Configure walk length based on graph size
    walk_len = 8 if _graph.number_of_nodes() > 500 else 10
    num_w = 10 if _graph.number_of_nodes() > 500 else 25
    
    n2v = train_node2vec(
        g_train,
        dimensions=16,
        walk_length=walk_len,
        num_walks=num_w,
        p=1.0,
        q=1.0,
        window=5,
        seed=seed,
        workers=1,
        quiet=True
    )
    embeddings = extract_all_embeddings(n2v, g_train)
    
    X_train, y_train = build_link_prediction_dataset(train_pos, train_neg, embeddings, operator="hadamard")
    clf = train_link_prediction_model(X_train, y_train, seed=seed)
    
    metadata = {
        "dataset_key": dataset_key,
        "model_type": "LogisticRegression",
        "operator": "hadamard",
        "dimensions": 16,
        "test_ratio": test_ratio,
        "seed": seed
    }
    
    # Save for future app startups
    try:
        save_embeddings(embeddings, emb_path)
        save_link_prediction_model(clf, metadata, clf_path)
    except Exception:
        pass
        
    return embeddings, clf, metadata


@st.cache_data(show_spinner="Computing algorithmic benchmarks...")
def get_cached_evaluation_dataframe(
    _graph: nx.Graph,
    _embeddings: Dict[Any, np.ndarray],
    _clf: LogisticRegression,
    dataset_key: str = "karate",
    k: int = 5
) -> pd.DataFrame:
    """Load or compute benchmark evaluation metrics."""
    if dataset_key == "facebook_ego0":
        processed_csv = Path("data") / "processed" / "stage7_facebook_evaluation_results.csv"
    else:
        processed_csv = Path("data") / "processed" / "model_evaluation_results.csv"
        
    if processed_csv.is_file():
        try:
            return pd.read_csv(processed_csv)
        except Exception:
            pass
            
    g_train, train_pos, test_pos = split_edges(_graph, test_ratio=0.2, seed=42)
    train_neg, test_neg = sample_negative_edges(_graph, len(train_pos), len(test_pos), seed=seed)
    
    return evaluate_all_models(
        graph_train=g_train,
        test_pos_edges=test_pos,
        test_neg_edges=test_neg,
        embeddings=_embeddings,
        model_lr=_clf,
        k=k,
        operator="hadamard"
    )


# ============================================================================
# Main Application
# ============================================================================

def main():
    st.set_page_config(
        page_title="Social Network Recommendation Engine",
        page_icon="🕸️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # -------------------------------------------------------------
    # Sidebar: Dataset Selection & User Controls
    # -------------------------------------------------------------
    st.sidebar.title("⚙️ Control Panel")
    
    st.sidebar.markdown("### 1. Dataset Selection")
    dataset_map = {
        "Zachary's Karate Club (34 nodes, 78 edges)": "karate",
        "SNAP Facebook - Ego Network 0 (348 nodes, 2,866 edges)": "facebook_ego0",
        "SNAP Facebook - Full Network (4,039 nodes, 88,234 edges)": "facebook_full"
    }
    selected_dataset_name = st.sidebar.selectbox(
        "Choose Social Network:",
        options=list(dataset_map.keys()),
        index=0,
        help="Switch between the classical Karate Club dataset and the real-world SNAP Facebook network."
    )
    dataset_key = dataset_map[selected_dataset_name]
    
    # Load chosen graph and models
    graph, stats = get_cached_graph_and_stats(dataset_key)
    embeddings, clf_model, model_meta = get_cached_models_and_embeddings(graph, dataset_key=dataset_key)
    
    st.sidebar.markdown("### 2. User Selection")
    all_nodes = sorted(list(graph.nodes()))
    
    # Format node labels with affiliation/degree
    def format_user_label(node_id: int) -> str:
        club = graph.nodes[node_id].get("club", "Member")
        deg = graph.degree(node_id)
        return f"User {node_id} ({deg} friends)" if club == "Member" else f"User {node_id} ({club}, {deg} friends)"
        
    # Provide a curated default node for convenience
    default_index = 0
    if dataset_key == "facebook_ego0" and 0 in all_nodes:
        default_index = all_nodes.index(0)
    elif dataset_key == "facebook_full" and 0 in all_nodes:
        default_index = all_nodes.index(0)
        
    selected_user = st.sidebar.selectbox(
        "Select Active User:",
        options=all_nodes,
        index=default_index,
        format_func=format_user_label,
        help="Select a social network member to generate friend recommendations."
    )
    
    st.sidebar.markdown("### 3. Recommendation Settings")
    top_k = st.sidebar.slider(
        "Number of Recommendations (Top-K):",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )
    
    algo_options = [
        "All Algorithms (Side-by-Side Comparison)",
        "Supervised Logistic Regression (Node2Vec + Hadamard)",
        "Node2Vec Embedding Cosine Similarity",
        "Common Neighbors (CN)",
        "Jaccard Similarity Coefficient",
        "Preferential Attachment (PA)",
    ]
    
    selected_algo = st.sidebar.selectbox(
        "Recommendation Algorithm:",
        options=algo_options,
        index=0,
        help="Choose a specific link prediction algorithm or compare all methods."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Active Dataset:** {selected_dataset_name.split('(')[0].strip()}  \n"
        f"**Total Nodes (|V|):** {stats['num_nodes']}  \n"
        f"**Total Edges (|E|):** {stats['num_edges']}  \n"
        f"**Model Embedding Dim:** 16"
    )
    
    # -------------------------------------------------------------
    # Section A: Home Dashboard & Network Overview
    # -------------------------------------------------------------
    st.title("🕸️ Social Network Recommendation Engine")
    st.caption("Graph Representation Learning, Topological Heuristics, and Supervised Link Prediction")
    
    # Metric KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Members (|V|)", stats["num_nodes"])
    col2.metric("Total Friendships (|E|)", stats["num_edges"])
    col3.metric("Network Density", f"{stats['density']:.4f}")
    col4.metric("Average Friends", f"{stats['average_degree']:.2f}")
    col5.metric("Avg Clustering", f"{stats['average_clustering_coefficient']:.3f}")
    
    # Selected User Overview Banner
    user_club = graph.nodes[selected_user].get("club", "Member")
    user_degree = graph.degree(selected_user)
    user_friends = sorted(list(graph.neighbors(selected_user)))
    user_candidates = sorted(list(get_candidate_users(graph, selected_user)))
    
    with st.expander(f"👤 Active User Profile: User {selected_user} ({user_degree} friends)", expanded=True):
        pcol1, pcol2 = st.columns([1, 2])
        with pcol1:
            st.markdown(f"- **User Identifier:** `Node {selected_user}`")
            if user_club != "Member":
                st.markdown(f"- **Faction Affiliation:** `{user_club}`")
            st.markdown(f"- **Direct Friends Count:** `{user_degree}`")
            st.markdown(f"- **Eligible Candidates:** `{len(user_candidates)} members`")
        with pcol2:
            st.markdown(f"**Current Friends List ({len(user_friends)}):**")
            sample_friends = user_friends[:30]
            friends_text = ", ".join([f"**User {f}**" for f in sample_friends])
            if len(user_friends) > 30:
                friends_text += f", ... (+{len(user_friends) - 30} more)"
            st.write(friends_text)
            
    # -------------------------------------------------------------
    # Compute Recommendations for Selected User
    # -------------------------------------------------------------
    with st.spinner("Computing recommendations across algorithms..."):
        # Traditional heuristics
        stage2_recs = recommend_all_heuristics(graph, selected_user, top_k=top_k)
        cn_recs = stage2_recs["common_neighbors"]
        jac_recs = stage2_recs["jaccard"]
        pa_recs = stage2_recs["preferential_attachment"]
        
        # Node2Vec Cosine
        n2v_recs = recommend_node2vec_cosine(graph, embeddings, selected_user, top_k=top_k)
        
        # Supervised LR
        sup_recs = recommend_supervised_link_prediction(
            graph, clf_model, embeddings, selected_user, top_k=top_k, operator="hadamard"
        )
        
    # Map recommendations to the active algorithm view
    current_recs: List[Dict[str, Any]] = []
    if "Supervised" in selected_algo:
        current_recs = sup_recs
        score_col_name = "P(Friendship)"
    elif "Cosine" in selected_algo:
        current_recs = n2v_recs
        score_col_name = "Cosine Similarity"
    elif "Common Neighbors" in selected_algo:
        current_recs = cn_recs
        score_col_name = "Common Neighbors"
    elif "Jaccard" in selected_algo:
        current_recs = jac_recs
        score_col_name = "Jaccard Index"
    elif "Preferential" in selected_algo:
        current_recs = pa_recs
        score_col_name = "Degree Product"
    else:
        # Default for side-by-side: use Supervised LR for graph highlighting
        current_recs = sup_recs
        score_col_name = "P(Friendship)"
        
    recommended_node_ids = [r["candidate_id"] for r in current_recs]
    
    # -------------------------------------------------------------
    # Navigation Tabs
    # -------------------------------------------------------------
    tab_recs, tab_comparison, tab_eval, tab_theory = st.tabs([
        "🎯 Friend Recommendations & Network Graph",
        "📊 Algorithmic Comparison",
        "📈 Evaluation Benchmark & Metrics",
        "🧠 Theory & Formulations"
    ])
    
    # -------------------------------------------------------------
    # TAB 1: Recommendations & Interactive Network Visualization
    # -------------------------------------------------------------
    with tab_recs:
        st.subheader(f"Recommendations for User {selected_user} via {selected_algo}")
        
        col_table, col_viz = st.columns([1, 1], gap="medium")
        
        with col_table:
            st.markdown(f"**Top-{top_k} Recommended Friends:**")
            
            table_rows = []
            for r in current_recs:
                cand_id = r["candidate_id"]
                cand_club = graph.nodes[cand_id].get("club", "Member")
                mutuals_str = ", ".join(map(str, r["mutual_friends"])) if r["mutual_friends"] else "None"
                
                row_dict = {
                    "Rank": f"#{r['rank']}",
                    "Candidate": f"User {cand_id}",
                    score_col_name: f"{r['score']:.4f}" if isinstance(r["score"], float) else str(r["score"]),
                    "Mutuals Count": r["num_mutual_friends"],
                    "Mutual Friends": mutuals_str,
                    "Total Friends": r["candidate_degree"]
                }
                if user_club != "Member":
                    row_dict["Affiliation"] = cand_club
                table_rows.append(row_dict)
                
            df_display = pd.DataFrame(table_rows)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            st.info(
                f"💡 **Recommendation Insight:** Evaluated candidate pool of {len(user_candidates)} eligible members. "
                f"All candidates strictly exclude the query user and existing friends."
            )
            
        with col_viz:
            st.markdown("**Interactive Social Network Visualization:**")
            if graph.number_of_nodes() > 350:
                st.caption(
                    "ℹ️ For large graphs, an **ego-neighborhood view** (query user, friends, and top candidates) "
                    "is dynamically rendered for responsive interaction."
                )
            fig = create_interactive_network_figure(
                graph,
                selected_user=selected_user,
                recommended_candidates=recommended_node_ids,
                title=f"Network View (User {selected_user} Highlighted)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    # -------------------------------------------------------------
    # TAB 2: Algorithmic Comparison
    # -------------------------------------------------------------
    with tab_comparison:
        st.subheader(f"Side-by-Side Comparison for User {selected_user}")
        st.markdown(
            "Comparing how different link prediction paradigms prioritize candidate friends for the same user:"
        )
        
        comp_rows = []
        for rank in range(1, top_k + 1):
            cn_i = cn_recs[rank - 1]
            jac_i = jac_recs[rank - 1]
            pa_i = pa_recs[rank - 1]
            n2v_i = n2v_recs[rank - 1]
            sup_i = sup_recs[rank - 1]
            
            comp_rows.append({
                "Rank": f"#{rank}",
                "Common Neighbors": f"User {cn_i['candidate_id']} ({cn_i['score']} mutuals)",
                "Jaccard": f"User {jac_i['candidate_id']} (J={jac_i['score']:.3f})",
                "Preferential Att.": f"User {pa_i['candidate_id']} (score={pa_i['score']})",
                "Node2Vec Cosine": f"User {n2v_i['candidate_id']} (cos={n2v_i['score']:.3f})",
                "Supervised LR": f"User {sup_i['candidate_id']} (P={sup_i['score']:.3f})",
            })
            
        df_comp = pd.DataFrame(comp_rows)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        
        st.markdown("### 🔍 Critical Takeaways from Comparison")
        st.markdown(
            "- **The Hub Bias of Traditional Heuristics:** Common Neighbors and Preferential Attachment "
            "often elevate the network's most connected nodes, regardless of local community affinity.\n"
            "- **Homophily Discovery in Embedding Methods:** Node2Vec and Supervised Logistic Regression "
            "prioritize candidates from the **same social cluster**, discovering multi-hop latent connections."
        )
        
    # -------------------------------------------------------------
    # TAB 3: Model Evaluation Benchmark
    # -------------------------------------------------------------
    with tab_eval:
        st.subheader(f"Model Evaluation Benchmark on Held-Out Test Edges ({selected_dataset_name.split('(')[0].strip()})")
        st.markdown(
            r"All algorithms were benchmarked under **identical, leak-free test conditions**: "
            r"evaluating on held-out positive test edges ($20\%$) and disjoint negative edge samples."
        )
        
        df_eval = get_cached_evaluation_dataframe(graph, embeddings, clf_model, dataset_key=dataset_key, k=5)
        st.dataframe(df_eval, use_container_width=True, hide_index=True)
        
        # Plot evaluation results
        eval_fig = go.Figure()
        algos = df_eval["Algorithm"].tolist()
        
        eval_fig.add_trace(go.Bar(
            name="ROC-AUC",
            x=algos,
            y=df_eval["ROC-AUC"],
            marker_color="#ff7f0e"
        ))
        eval_fig.add_trace(go.Bar(
            name="Average Precision",
            x=algos,
            y=df_eval["Average Precision"],
            marker_color="#9467bd"
        ))
        eval_fig.add_trace(go.Bar(
            name="Precision@5",
            x=algos,
            y=df_eval["Precision@5"],
            marker_color="#2ca02c"
        ))
        eval_fig.add_trace(go.Bar(
            name="Recall@5",
            x=algos,
            y=df_eval["Recall@5"],
            marker_color="#1f77b4"
        ))
        
        eval_fig.update_layout(
            title=f"Benchmark Metrics Comparison ({selected_dataset_name.split('(')[0].strip()})",
            barmode="group",
            yaxis=dict(title="Score", range=[0.0, 1.0]),
            xaxis=dict(tickangle=-20),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            margin=dict(l=10, r=10, t=50, b=40)
        )
        st.plotly_chart(eval_fig, use_container_width=True)
        
        st.warning(
            "⚠️ **Academic Caveat:** In real-world social platforms, graph density is extremely low "
            "($\\rho \\sim 10^{-6}$). Real systems face extreme class imbalance, requiring models "
            "to filter thousands of candidate negatives for every true link."
        )
        
    # -------------------------------------------------------------
    # TAB 4: Theory & Mathematical Formulations
    # -------------------------------------------------------------
    with tab_theory:
        st.subheader("Mathematical Foundations & Dataset Provenance")
        
        st.markdown("#### 📁 Dataset Provenance & Proven Properties")
        st.markdown(
            "- **Stanford SNAP Facebook Social Circles (McAuley & Leskovec, NIPS 2012):** "
            "Contains 4,039 survey participants and 88,234 mutual friendships collected from 10 Facebook ego networks. "
            "*Density:* 0.0108 (~1.08%). *Average Degree:* 43.69. *Average Clustering:* 0.6055.\n"
            "- **Zachary's Karate Club (Zachary, 1977):** "
            "Canonical 34-member social fission network. *Density:* 0.1390. *Average Degree:* 4.59."
        )
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("#### 1. Common Neighbors (CN)")
            st.latex(r"CN(u, v) = |N(u) \cap N(v)|")
            st.caption("Counts shared mutual acquaintances. Explores 1-hop neighborhood homophily.")
            
            st.markdown("#### 2. Jaccard Similarity Coefficient")
            st.latex(r"J(u, v) = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}")
            st.caption("Normalizes overlap by total combined social footprint, mitigating hub bias.")
            
            st.markdown("#### 3. Preferential Attachment (PA)")
            st.latex(r"PA(u, v) = \text{deg}(u) \times \text{deg}(v)")
            st.caption("Models scale-free rich-get-richer attachment. Ranks candidates by global popularity.")
            
        with m_col2:
            st.markdown("#### 4. Node2Vec Biased Random Walks")
            st.latex(r"\pi_{vx} = \alpha_{pq}(t, x) \cdot w_{vx}")
            st.markdown(
                "- **Return parameter $p$:** Controls likelihood of immediately backtracking.\n"
                "- **In-out parameter $q$:** Controls BFS (homophily) vs DFS (structural roles)."
            )
            
            st.markdown("#### 5. Supervised Link Prediction")
            st.latex(r"\mathbf{x}_{(u, v)} = \mathbf{z}_u \odot \mathbf{z}_v \quad (\text{Hadamard Product})")
            st.latex(r"P(\text{Link} \mid u, v) = \sigma(\mathbf{w}^T \mathbf{x}_{(u, v)} + b)")
            st.caption("Trains Logistic Regression on symmetric Hadamard edge features from leak-free training embeddings.")


if __name__ == "__main__":
    main()
