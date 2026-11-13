# Social Network Recommendation Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5%2B-3C2179.svg)](https://pyg.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.2-orange.svg)](https://networkx.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-FF4B4B.svg)](https://streamlit.io/)
[![Unit Tests](https://img.shields.io/badge/tests-67%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Graph Machine Learning system that predicts missing edges and recommends social connections by bridging **classical graph topology**, **shallow representation learning (Node2Vec)**, and **deep message-passing Graph Neural Networks (GCN & GraphSAGE)**. Features an interactive **Streamlit** dashboard, strict leak-free evaluation pipelines, and multi-scale validation across **Zachary's Karate Club** and **Stanford SNAP Facebook Social Circles**.

> 📖 **Theory & Conceptual Guide:** For a simple, intuitive A-to-Z walkthrough of the project's logic, model mechanics, datasets, metrics, and presentation Q&A, see the [Theory & Implementation Guide](docs/THEORY_AND_IMPLEMENTATION_GUIDE.md).
>
> 📄 **Academic Project Report (PDF):** Download the concise 3-page summary report: [Social_Network_Recommendation_Report.pdf](docs/Social_Network_Recommendation_Report.pdf) (Authors: **N S Balaji** - RA2311056010012, **Yuvraj** - RA2311056010026).

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Live Web Application & Sample Outputs](#-live-web-application--sample-outputs)
  - [Tab 1: Top-5 Friend Recommendations](#sample-output-1-top-5-friend-recommendations-tab-1)
  - [Tab 2: Side-by-Side Algorithmic Comparison](#sample-output-2-side-by-side-algorithmic-comparison-tab-2)
  - [Tab 3: SNAP Facebook Large-Scale Recommendations](#sample-output-3-real-world-snap-facebook-recommendations-user-9)
- [Algorithmic Formulations](#-algorithmic-formulations)
  - [1. Topological Heuristics](#1-topological-heuristics)
  - [2. Node2Vec Representation Learning](#2-node2vec-representation-learning)
  - [3. Supervised Edge Classification](#3-supervised-edge-classification)
  - [4. Graph Neural Networks (GCN & GraphSAGE)](#4-graph-neural-networks-gcn--graphsage)
- [Empirical Benchmarks & Findings](#-empirical-benchmarks--findings)
  - [Karate Club Benchmark](#karate-club-benchmark-held-out-test-set)
  - [SNAP Facebook Ego-0 Benchmark](#snap-facebook-ego-0-benchmark)
  - [Qualitative Analysis: Resolving Hub Bias](#qualitative-analysis-resolving-hub-bias)
- [Project Directory Layout](#-project-directory-layout)
- [Installation & Setup](#-installation--setup)
- [Usage & Execution Guide](#-usage--execution-guide)
  - [Interactive Web Application](#running-the-interactive-streamlit-application)
  - [Running Modular Pipeline Scripts](#running-modular-pipeline-scripts)
  - [Running the Automated Test Suite](#running-the-automated-test-suite)
- [Academic Caveats & Limitations](#-academic-caveats--limitations)
- [References & Citations](#-references--citations)

---

## 🔬 Executive Overview

Friend recommendation is a core task in modern social networks, conventionally framed as **Link Prediction**. Given a snapshot of a social graph $G = (V, E)$, the objective is to predict whether an unobserved edge $(u, v) \notin E$ is likely to form in the future.

This project implements a complete, leak-free machine learning workflow:
1. **Bridge-Preserving Edge Splitting**: Partitions graph edges into train ($80\%$) and test ($20\%$) subsets while enforcing graph connectivity (`nx.is_connected(g_train)`), preventing test link leakage during random walks or message passing.
2. **Disjoint Negative Sampling**: Accurately samples non-existent links $(u, v) \notin E$ ensuring zero intersection between train and test negatives.
3. **Multi-Paradigm Algorithmic Comparison**: Benchmarks 7 algorithms ranging from local heuristics ($O(1)$) to deep message-passing architectures ($O(|V| \cdot d)$).
4. **Multi-Scale Real-World Scaling**: Benchmarked across the canonical Karate Club ($|V|=34, |E|=78$) and Stanford SNAP Facebook ($|V|=4,039, |E|=88,234$).
5. **Interactive UI**: Web application with real-time Plotly graph visualization and ego-subgraph extraction for large-scale graphs.

---

## 🏛️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION LAYER                          │
│   Zachary's Karate Club (|V|=34)   │   Stanford SNAP Facebook (|V|=4,039) │
└───────────────────┬───────────────────────────────────┬────────────────┘
                    │                                   │
                    ▼                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   GRAPH BUILDER & LEAK-FREE SPLIT                      │
│   • Bridge-preserving edge split (80% Train, 20% Held-out Test)       │
│   • Balanced negative edge sampling (Disjoint non-edges)               │
└───────────────┬───────────────────────────────┬────────────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────────┐   ┌────────────────────────────────────┐
│      TOPOLOGICAL HEURISTICS   │   │     GRAPH REPRESENTATION LEARNING  │
│   • Common Neighbors (CN)     │   │   • Biased 2nd-Order Walks (p, q)  │
│   • Jaccard Similarity        │   │   • Skip-Gram Word2Vec (d=16)      │
│   • Preferential Attachment   │   │   • Cosine Similarity Scoring      │
└───────────────┬───────────────┘   └─────────────────┬──────────────────┘
                │                                     │
                │                   ┌─────────────────┴──────────────────┐
                │                   ▼                                    ▼
                │   ┌──────────────────────────────┐     ┌───────────────┴──────────────┐
                │   │    SUPERVISED EDGE ML        │     │  DEEP GRAPH NEURAL NETWORKS  │
                │   │  • Hadamard Edge Features    │     │  • 2-Layer GCNConv           │
                │   │  • L2-Regularized LogReg     │     │  • 2-Layer SAGEConv (Mean)   │
                │   │  • Calibrated Probabilities  │     │  • Inner-Product Link Decoder│
                │   └──────────────┬───────────────┘     └───────────────┬──────────────┘
                │                  │                                     │
                ▼                  ▼                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│               UNIFIED BENCHMARK & EVALUATION ENGINE                    │
│   • Classification: ROC-AUC, Average Precision (PR-AUC)                │
│   • Ranking: Precision@K, Recall@K                                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              INTERACTIVE STREAMLIT APPLICATION (app.py)                │
│   • Dynamic multi-dataset selector & Active user profile               │
│   • Interactive Plotly 2D graph with ego-subgraph rendering            │
│   • Real-time candidate recommendation ranking & Metric dashboard      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Live Web Application & Sample Outputs

The interactive web dashboard (`app.py`) provides an intuitive graphical interface for social network exploration, recommendation ranking, and multi-algorithm evaluation.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ⚙️ Control Panel          │ 🕸️ Social Network Recommendation Engine                     │
│ 1. Dataset Selection     │ ────────────────────────────────────────────────────────── │
│    [Karate Club ▼]       │ Members: 34 │ Friendships: 78 │ Density: 0.1390 │ Avg Deg: 4.59│
│ 2. Active User Selector  │ ────────────────────────────────────────────────────────── │
│    [User 0 (16 friends)▼]│ 👤 User 0 Profile: Faction 'Mr. Hi' | 16 Friends | 17 Cand. │
│ 3. Recommendations (K)   │ ┌────────────────────────────────────────────────────────┐ │
│    [───●──────] 5        │ │ 🎯 Recommendations │ 📊 Comparison │ 📈 Benchmark │ 🧠 │ │
│ 4. Model Paradigm        │ └────────────────────────────────────────────────────────┘ │
│    [Supervised LR ▼]     │ [ Top-5 Candidate Table ]     [ Interactive Plotly Graph ] │
└──────────────────────────┴────────────────────────────────────────────────────────────┘
```

---

### Sample Output 1: Top-5 Friend Recommendations (Tab 1)

When **User 0 (Faction 'Mr. Hi')** is selected on **Zachary's Karate Club**, the system filters out all 16 existing friends and the query user itself, scoring all 17 eligible candidates. Below is the live tabular output from the dashboard:

| Rank | Candidate | Affiliation | P(Friendship) | Mutuals Count | Mutual Friends | Candidate Total Friends |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **User 16** | Mr. Hi | **0.7061** | 2 | 5, 6 | 2 |
| **#2** | **User 9** | Officer | **0.3912** | 1 | 2 | 2 |
| **#3** | **User 28** | Mr. Hi | **0.3204** | 2 | 2, 33 | 3 |
| **#4** | **User 15** | Officer | **0.3114** | 0 | None | 2 |
| **#5** | **User 20** | Officer | **0.2981** | 0 | None | 2 |

> 💡 **Recommendation Insight:** In the web application, the interactive Plotly 2D canvas highlights User 0 in **bright gold**, direct connections in **blue**, and draws dashed **green connection lines** to recommended candidates.

---

### Sample Output 2: Side-by-Side Algorithmic Comparison (Tab 2)

Comparing how each link prediction paradigm ranks candidate friends for the same active user:

| Rank | Common Neighbors | Jaccard Similarity | Preferential Attachment | Node2Vec Cosine | Supervised LR | GCN Link Predictor |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | User 33 (4 mutuals) ⚠️ | User 33 (J=0.138) ⚠️ | User 33 (deg product: 272) ⚠️ | **User 16 (cos: 0.951)** | **User 16 (P: 0.706)** | **User 30 (P: 1.000)** |
| **#2** | User 32 (3 mutuals) ⚠️ | User 16 (J=0.125) | User 32 (deg product: 192) ⚠️ | **User 9 (cos: 0.703)** | **User 9 (P: 0.391)** | **User 16 (P: 0.999)** |
| **#3** | User 16 (2 mutuals) | User 32 (J=0.120) | User 23 (deg product: 80) | **User 28 (cos: 0.630)** | **User 28 (P: 0.320)** | **User 28 (P: 0.994)** |
| **#4** | User 28 (2 mutuals) | User 28 (J=0.118) | User 27 (deg product: 64) | User 20 (cos: 0.570) | User 15 (P: 0.311) | User 26 (P: 0.001) |
| **#5** | User 30 (2 mutuals) | User 30 (J=0.111) | User 29 (deg product: 64) | User 15 (cos: 0.547) | User 20 (P: 0.298) | User 20 (P: 0.000) |

> ⚠️ **The Critical Takeaway (Resolving Hub Bias):**  
> Notice that simple heuristics (Common Neighbors and Preferential Attachment) rank **User 33** (the opposing faction leader) at Rank #1 because User 33 is a high-degree celebrity hub (17 friends). Recommending a rival leader violates community homophily!  
> In contrast, **Node2Vec, Supervised LR, and GCN** correctly identify latent community structure, assigning near-zero probability to User 33 and prioritizing **User 16, User 28, and User 30** (fellow members of Mr. Hi's faction).

---

### Sample Output 3: Real-World SNAP Facebook Recommendations (User 9)

On the **SNAP Facebook Ego-0** dataset (348 nodes, 2,866 friendships), selecting **User 9** (57 friends, 290 eligible candidates):

| Rank | Candidate | Friendship Probability | Mutual Friends Count | Sample Mutual Friends | Candidate Degree |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **User 62** | **0.9614** (96.1%) | 18 mutuals | 0, 26, 56, 67, 122... | 26 friends |
| **#2** | **User 10** | **0.9548** (95.5%) | 9 mutuals | 0, 67, 142, 169, 200... | 10 friends |
| **#3** | **User 183** | **0.9499** (95.0%) | 2 mutuals | 0, 133 | 2 friends |
| **#4** | **User 274** | **0.9439** (94.4%) | 12 mutuals | 0, 3, 21, 75, 85... | 14 friends |
| **#5** | **User 123** | **0.9351** (93.5%) | 11 mutuals | 0, 21, 26, 67, 122... | 18 friends |

---

## 📐 Algorithmic Formulations

### 1. Topological Heuristics

- **Common Neighbors (CN):**
  $$CN(u, v) = |N(u) \cap N(v)|$$
  Measures the number of shared mutual acquaintances between $u$ and $v$ (1-hop triadic closure).

- **Jaccard Similarity Coefficient:**
  $$J(u, v) = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}$$
  Normalizes neighborhood intersection by the total union footprint, penalizing high-degree hubs.

- **Preferential Attachment (PA):**
  $$PA(u, v) = \text{deg}(u) \times \text{deg}(v)$$
  Models the rich-get-richer growth phenomenon in scale-free networks based purely on node degree.

---

### 2. Node2Vec Representation Learning
Grover & Leskovec (KDD 2016) proposed learning continuous mappings $\mathbf{Z} \in \mathbb{R}^{|V| \times d}$ via flexible 2nd-order random walks parameterized by $p$ and $q$:
$$\pi_{vx} = \alpha_{pq}(t, x) \cdot w_{vx}$$
Where transition bias $\alpha_{pq}(t, x)$ for walk edge $(t, v)$ moving to neighbor $x$ is:
$$\alpha_{pq}(t, x) = \begin{cases} 
\frac{1}{p} & \text{if } d_{tx} = 0 \text{ (backtrack to previous node)} \\
1 & \text{if } d_{tx} = 1 \text{ (local neighborhood / homophily)} \\
\frac{1}{q} & \text{if } d_{tx} = 2 \text{ (outward structural exploration)}
\end{cases}$$
Node sequences are optimized using Skip-gram with negative sampling, and pairwise proximity is calculated via cosine similarity:
$$\text{Cosine}(u, v) = \frac{\mathbf{z}_u \cdot \mathbf{z}_v}{\|\mathbf{z}_u\|_2 \|\mathbf{z}_v\|_2}$$

---

### 3. Supervised Edge Classification
To classify edge formation, continuous node embeddings are mapped to edge features using the **Hadamard (element-wise) operator**:
$$\mathbf{x}_{(u, v)} = \mathbf{z}_u \odot \mathbf{z}_v \in \mathbb{R}^d, \quad x_{(u, v), i} = z_{u, i} \cdot z_{v, i}$$

> **Commutative Invariance:** Because $z_{u, i} \cdot z_{v, i} \equiv z_{v, i} \cdot z_{u, i}$, the Hadamard operator produces identical feature vectors regardless of edge node query order $(u, v) \equiv (v, u)$, making it strictly suitable for undirected networks.

A regularized Logistic Regression classifier models edge probability:
$$P(\text{Link} \mid u, v) = \sigma(\mathbf{w}^\top \mathbf{x}_{(u, v)} + b) = \frac{1}{1 + \exp\left(-(\mathbf{w}^\top \mathbf{x}_{(u, v)} + b)\right)}$$

---

### 4. Graph Neural Networks (GCN & GraphSAGE)

#### Graph Convolutional Network (GCN)
Kipf & Welling (ICLR 2017) localized spectral graph convolutions with symmetric normalization:
$$\mathbf{H}^{(l+1)} = \sigma \left( \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$
- $\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_N$: Adjacency matrix augmented with self-loops.
- $\mathbf{\tilde{D}}_{ii} = \sum_j \mathbf{\tilde{A}}_{ij}$: Diagonal degree matrix.
- $\mathbf{\tilde{D}}^{-1/2}\mathbf{\tilde{A}}\mathbf{\tilde{D}}^{-1/2}$: Normalization preventing feature explosion in high-degree nodes.

#### GraphSAGE (Sample and Aggregate)
Hamilton et al. (NeurIPS 2017) defined inductive spatial neighborhood aggregation:
$$\mathbf{h}_v^{(l)} = \sigma \left( \mathbf{W}^{(l)} \cdot \left[ \mathbf{h}_v^{(l-1)} \,\|\, \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l-1)} \right] \right)$$

#### Inner-Product Link Decoder
Given node representations $\mathbf{z}_u, \mathbf{z}_v \in \mathbb{R}^d$ from the final GNN layer, link probability is decoded as:
$$\hat{y}_{uv} = \sigma(\mathbf{z}_u^\top \mathbf{z}_v) = \frac{1}{1 + \exp(-\mathbf{z}_u^\top \mathbf{z}_v)}$$
Trained end-to-end via Binary Cross-Entropy loss across positive edges and negative non-edges.

---

## 📊 Empirical Benchmarks & Findings

### Karate Club Benchmark (Held-out Test Set)

| Algorithm | Category | ROC-AUC | Average Precision | Precision@5 | Recall@5 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Common Neighbors** | Topological Heuristic | 0.6333 | 0.5784 | 0.1250 | 0.5000 |
| **Jaccard Similarity** | Topological Heuristic | 0.6067 | 0.5445 | 0.1250 | 0.5000 |
| **Preferential Attachment** | Topological Heuristic | 0.6733 | 0.6794 | 0.1375 | 0.5333 |
| **Node2Vec Cosine** | Shallow Representation | 0.7422 | 0.7033 | 0.1250 | 0.5000 |
| **Node2Vec + Logistic Regression** | Supervised Shallow | 0.7822 | 0.7676 | 0.1250 | 0.5000 |
| **GraphSAGE Link Predictor** | Graph Neural Network | 0.7333 | 0.6775 | — | — |
| **GCN Link Predictor** | **Graph Neural Network** | **0.8311** | **0.8769** | — | — |

---

### SNAP Facebook Ego-0 Benchmark

| Algorithm | Category | ROC-AUC | Average Precision | Precision@5 | Recall@5 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Common Neighbors** | Topological Heuristic | **0.9368** | **0.9298** | 0.3556 | 0.2227 |
| **Jaccard Similarity** | Topological Heuristic | 0.8989 | 0.8809 | 0.3012 | 0.1852 |
| **Preferential Attachment** | Topological Heuristic | 0.6346 | 0.6121 | 0.1444 | 0.0874 |
| **Node2Vec Cosine** | Shallow Representation | 0.8780 | 0.8864 | 0.3125 | 0.1983 |
| **Node2Vec + Logistic Regression** | Supervised Shallow | 0.8802 | 0.8879 | 0.3125 | 0.1983 |

---

### Qualitative Analysis: Resolving Hub Bias

Examining the top recommended friends for **User 0 (Faction 'Mr. Hi')** illustrates a critical distinction between heuristics and representation learning:

```text
Target User: User 0 (Mr. Hi's Faction)
-----------------------------------------------------------------------------------------
Rank │ Common Neighbors       │ Node2Vec Cosine        │ GCN Link Predictor
─────┼────────────────────────┼────────────────────────┼────────────────────────────────
 1   │ User 33 (4 mutuals) ⚠️ │ User 16 (score: 0.951) │ User 30 (score: 1.000)
 2   │ User 32 (3 mutuals) ⚠️ │ User 9  (score: 0.703) │ User 16 (score: 0.999)
 3   │ User 16 (2 mutuals)    │ User 28 (score: 0.630) │ User 28 (score: 0.994)
 4   │ User 28 (2 mutuals)    │ User 20 (score: 0.570) │ User 26 (score: 0.001)
 5   │ User 30 (2 mutuals)    │ User 15 (score: 0.547) │ User 20 (score: 0.000)
-----------------------------------------------------------------------------------------
```

- **Topological Hub Bias**: Common Neighbors recommends **User 33** (the officer of the opposing faction) at Rank 1 simply because User 33 is a global hub with 17 connections who happens to share peripheral mutual friends. In production, recommending rival leaders degrades user trust.
- **Latent Homophily Discovery**: Both Node2Vec and GCN assign near-zero probability to User 33 and prioritize **User 16**, **User 30**, and **User 28**—all loyal members of User 0's community. Multi-hop representations capture community structure beyond raw degree counts.

---

## 📁 Project Directory Layout

```text
social-network-recommendation/
│
├── data/
│   ├── raw/                           # Raw edge lists (karate_club_edges.csv, facebook_combined.txt)
│   └── processed/                     # Preprocessed splits, stats JSON, and benchmark CSVs
│
├── models/                            # Trained models and checkpoints
│   ├── node2vec_karate_embeddings.pkl # Node2Vec Karate node vectors (d=16)
│   ├── link_prediction_logistic_regression.pkl # Trained Logistic Regression classifier
│   ├── node2vec_facebook_ego0_embeddings.pkl   # Node2Vec Facebook Ego-0 vectors
│   ├── gcn_link_predictor.pt          # PyTorch GCN model weights
│   └── graphsage_link_predictor.pt    # PyTorch GraphSAGE model weights
│
├── reports/
│   └── figures/                       # Publication-grade evaluation figures
│       ├── karate_club_network.png    # Ground-truth Karate network visualization
│       ├── model_evaluation_comparison.png     # 5-way baseline evaluation plots
│       ├── facebook_evaluation_comparison.png  # SNAP Facebook benchmark comparison
│       └── gnn_vs_shallow_embeddings.png       # GNN vs Shallow embeddings comparison
│
├── scripts/                           # Standalone executable driver scripts
│   ├── run_stage1.py                  # Graph acquisition & topological analysis
│   ├── run_stage2.py                  # Traditional heuristic recommendations
│   ├── run_stage3.py                  # Node2Vec embedding training & cosine ranking
│   ├── run_stage4.py                  # Supervised Hadamard link classification
│   ├── run_stage5.py                  # Rigorous 5-way model evaluation benchmark
│   ├── run_stage7.py                  # SNAP Facebook downloading & multi-scale evaluation
│   └── run_stage8.py                  # GNN (GCN & GraphSAGE) link prediction benchmark
│
├── src/                               # Modular core library
│   ├── __init__.py                    # Library version (1.0.0)
│   ├── data_loader.py                 # NetworkX dataset loaders (Karate & SNAP Facebook)
│   ├── graph_builder.py               # Graph validation, edge splits, and edge-list exports
│   ├── graph_analysis.py              # Network statistics & Plotly graph visualization
│   ├── recommendation.py              # CN, Jaccard, Preferential Attachment heuristics
│   ├── embeddings.py                  # Biased random walks & Word2Vec Skip-gram pipeline
│   ├── link_prediction.py             # Hadamard edge features, negative sampling, LogReg
│   ├── evaluation.py                  # ROC-AUC, Average Precision, Precision@K, Recall@K
│   └── gnn.py                         # PyTorch Geometric GCN & GraphSAGE architectures
│
├── tests/                             # Comprehensive automated test suite (66 tests)
│   ├── test_environment.py            # Environment & dependency checks (4 tests)
│   ├── test_graph.py                  # Graph construction & analysis tests (12 tests)
│   ├── test_recommendation.py         # Heuristic recommendation unit tests (9 tests)
│   ├── test_embeddings.py             # Node2Vec walk & embedding tests (9 tests)
│   ├── test_link_prediction.py        # Edge splitting, sampling, and classifier tests (7 tests)
│   ├── test_evaluation.py             # Evaluation metric computation tests (6 tests)
│   ├── test_app.py                    # Streamlit component and loader tests (5 tests)
│   ├── test_snap_facebook.py          # SNAP Facebook parsing and ego-network tests (6 tests)
│   └── test_gnn.py                    # PyG GCN & GraphSAGE architecture tests (8 tests)
│
├── app.py                             # Interactive Streamlit Web Dashboard
├── requirements.txt                   # Pinned production dependencies
├── README.md                          # Comprehensive project documentation
└── .gitignore                         # Git exclusion rules
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.11** (tested on 3.11.9)
- **Git**

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NS145/Social-Network-Recommendation.git
   cd Social-Network-Recommendation
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     py -3.11 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3.11 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify the environment:**
   ```bash
   pytest tests/
   ```

---

## 🚀 Usage & Execution Guide

### Running the Interactive Streamlit Application
Launch the web application with real-time network visualization:
```bash
streamlit run app.py
```
*Access the dashboard at `http://localhost:8501`.*

**Interactive Features:**
- **Dataset Dropdown:** Switch dynamically between *Zachary's Karate Club*, *SNAP Facebook Ego-0*, and *SNAP Facebook Full*.
- **Active User Selector:** Inspect direct friends and calculate candidate recommendations for any network member.
- **Recommendation Slider:** Adjust Top-K ($K \in [3, 10]$).
- **Algorithm Selector:** Choose individual heuristics, Node2Vec Cosine, Supervised Logistic Regression, or GCN.
- **Dynamic Subgraph Extraction:** Automatically isolates ego-neighborhoods for large graphs ($|V| > 350$) to ensure high-performance browser rendering.

---

### Running Modular Pipeline Scripts

Execute any pipeline stage independently:

```bash
# Stage 1: Load Karate Club, compute network statistics, export edge-list
python scripts/run_stage1.py

# Stage 2: Compute heuristic recommendations (CN, Jaccard, PA)
python scripts/run_stage2.py

# Stage 3: Train Node2Vec embeddings and rank via cosine similarity
python scripts/run_stage3.py

# Stage 4: Train Supervised Logistic Regression with Hadamard edge features
python scripts/run_stage4.py

# Stage 5: Run comprehensive 5-way benchmark evaluation
python scripts/run_stage5.py

# Stage 7: Download and benchmark SNAP Facebook Social Circles
python scripts/run_stage7.py

# Stage 8: Train and evaluate GNNs (GCN & GraphSAGE) link predictors
python scripts/run_stage8.py
```

---

### Running the Automated Test Suite

Run all 66 automated unit tests across the codebase:
```bash
pytest tests/ -v
```

---

## ⚠️ Academic Caveats & Limitations

1. **Graph Density & Class Imbalance:**
   - In benchmark networks like Zachary's Karate Club ($\rho = 0.139$) or Facebook Ego-0 ($\rho = 0.047$), edge density is relatively high.
   - In production platforms (e.g., LinkedIn, Instagram), network density is extremely sparse ($\rho \sim 10^{-6}$). The ratio of true links to non-links is heavily imbalanced, requiring advanced negative sampling strategies (e.g., hard negative mining).
2. **Transductive vs. Inductive Representations:**
   - Node2Vec embeddings are transductive: new members cannot be embedded without generating fresh random walks.
   - Message-passing GNNs (GraphSAGE) are inductive and can infer embeddings for unseen nodes given input attributes.
3. **Temporal Dynamics:**
   - Real-world friendships evolve over continuous time. This model evaluates static snapshots; incorporating temporal edge timestamps would further enhance prediction fidelity.

---

## 📚 References & Citations

1. **Zachary's Karate Club:**
   - Zachary, W. W. (1977). *An Information Flow Model for Conflict and Fission in Small Groups.* Journal of Anthropological Research, 33(4), 452-473.
2. **Node2Vec:**
   - Grover, A., & Leskovec, J. (2016). *node2vec: Scalable Feature Learning for Networks.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16), 855-864.
3. **Graph Convolutional Networks (GCN):**
   - Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* International Conference on Learning Representations (ICLR '17).
4. **GraphSAGE:**
   - Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* Advances in Neural Information Processing Systems (NeurIPS '17), 1024-1034.
5. **SNAP Facebook Dataset:**
   - McAuley, J., & Leskovec, J. (2012). *Learning to Discover Social Circles in Ego Networks.* Advances in Neural Information Processing Systems (NeurIPS '12), 539-547.

---

## 👥 Authors
- **N S Balaji** (Registration No: `RA2311056010012`)
- **Yuvraj** (Registration No: `RA2311056010026`)

*SRM Institute of Science and Technology — Applied Machine Learning Project*

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
