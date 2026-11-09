# Social Network Recommendation Engine

A Graph Machine Learning application that recommends potential friends for a selected user based on social network topology, node representation learning (Node2Vec), and supervised link prediction.

---

## 📌 Project Overview

This project implements a multi-stage graph recommendation engine:
1. **Graph Construction & Structural Analysis:** Modeling social networks as undirected graphs and extracting topological properties.
2. **Traditional Heuristic Link Prediction:** Baseline recommendation algorithms based on neighborhood overlap (Common Neighbors, Jaccard Similarity, Preferential Attachment).
3. **Graph Representation Learning (Node2Vec):** Generating low-dimensional continuous node embeddings via biased random walks and Skip-gram architecture.
4. **Supervised Link Prediction:** Training classifiers (e.g., Logistic Regression) on element-wise edge embeddings (Hadamard product) with negative edge sampling to predict new friendships.
5. **Model Evaluation & Ranking:** Rigorous evaluation using Precision@K, Recall@K, ROC-AUC, and Average Precision.
6. **Interactive Streamlit Interface:** User-facing dashboard for network visualization, friend recommendation ranking, and metric comparison.
7. **Real-world Scaling:** Integration with the SNAP Facebook Social Circles dataset.
8. **Advanced Extension:** Optional Graph Neural Network (GNN) explorations.

---

## 📁 Project Architecture

```text
social-network-recommendation/
│
├── data/
│   ├── raw/                 # Raw datasets (e.g., edge lists, SNAP files)
│   └── processed/           # Preprocessed graph data, edge splits, train/test sets
│
├── models/                  # Serialized models, Node2Vec embeddings, checkpoints
│
├── notebooks/               # Exploratory data analysis and experimental notebooks
│
├── src/                     # Core modular Python package
│   ├── __init__.py          # Package initialization and versioning
│   ├── data_loader.py       # Dataset acquisition and raw file loading
│   ├── graph_builder.py     # Graph construction, edge validation, and splits
│   ├── graph_analysis.py    # Topological metrics (degree, density, clustering, etc.)
│   ├── recommendation.py    # Traditional graph-based recommendation heuristics
│   ├── embeddings.py        # Node2Vec representation learning pipeline
│   ├── link_prediction.py   # Feature engineering, negative sampling, and classification
│   └── evaluation.py        # Performance metrics (Precision@K, Recall@K, ROC-AUC)
│
├── tests/                   # Automated unit and integration test suite
│   ├── __init__.py
│   └── test_environment.py  # Environment verification and dependency sanity checks
│
├── app.py                   # Interactive Streamlit web application
├── requirements.txt         # Project dependencies with compatible version constraints
├── README.md                # Project documentation and stage-wise progress
└── .gitignore               # Git exclusion rules
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11 (verified compatibility with scientific graph learning stack)
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd social-network-recommendation
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

4. **Verify installation:**
   ```bash
   pytest tests/
   ```

---

## 🚀 Development Status

- [x] **Stage 0: Project Initialization & Environment Setup**
- [x] **Stage 1: Dataset Acquisition & Graph Construction**
- [x] **Stage 2: Traditional Graph-based Recommendation Algorithms**
- [x] **Stage 3: Node2Vec & Node Embeddings** (Completed)
  - Biased 2nd-order random walks with flexible return (p) and in-out (q) parameters
  - Continuous low-dimensional node representation learning via Skip-gram Word2Vec
  - Serialization of trained models and embedding dictionaries in `models/`
  - Cosine similarity ranking engine with strict candidate filtering
  - Side-by-side comparison illustrating homophily vs. 1-hop heuristic hub bias
  - 34 automated unit tests passing
- [ ] **Stage 4: Supervised Link Prediction**
- [ ] **Stage 5: Model Evaluation & Comparison**
- [ ] **Stage 6: Streamlit Application**
- [ ] **Stage 7: Real-world Dataset Integration (SNAP Facebook)**
- [ ] **Stage 8: Optional GNN Extension**

