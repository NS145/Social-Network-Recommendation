"""Environment and Dependency Verification Test.
Ensures that all required libraries can be imported and basic operations succeed.
"""

import sys
import pytest


def test_python_version():
    """Verify that Python version is 3.11 or compatible."""
    major, minor = sys.version_info.major, sys.version_info.minor
    assert major == 3, f"Expected Python 3, got Python {major}"
    assert minor >= 10, f"Expected Python 3.10+, got Python {major}.{minor}"


def test_imports():
    """Verify that all core project dependencies import without errors."""
    import numpy as np
    import scipy
    import pandas as pd
    import networkx as nx
    import sklearn
    import gensim
    import node2vec
    import streamlit
    import plotly
    import matplotlib

    assert np.__version__ is not None
    assert scipy.__version__ is not None
    assert pd.__version__ is not None
    assert nx.__version__ is not None
    assert sklearn.__version__ is not None
    assert gensim.__version__ is not None
    assert node2vec.__version__ is not None
    assert streamlit.__version__ is not None
    assert plotly.__version__ is not None
    assert matplotlib.__version__ is not None


def test_src_package_structure():
    """Verify that all modular src packages exist and are importable."""
    import src
    from src import data_loader
    from src import graph_builder
    from src import graph_analysis
    from src import recommendation
    from src import embeddings
    from src import link_prediction
    from src import evaluation
    from src import gnn

    assert src.__version__ == "1.0.0"
    assert data_loader is not None
    assert graph_builder is not None
    assert graph_analysis is not None
    assert recommendation is not None
    assert embeddings is not None
    assert link_prediction is not None
    assert evaluation is not None
    assert gnn is not None


def test_networkx_basic_functionality():
    """Verify that NetworkX can instantiate and query an undirected graph."""
    import networkx as nx

    g = nx.Graph()
    g.add_edge(1, 2)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    assert not g.is_directed()

