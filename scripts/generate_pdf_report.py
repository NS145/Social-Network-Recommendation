"""Script to generate a publication-quality 3-page PDF summary report for the project.
Includes student names: N S Balaji (RA2311056010012) & Yuvraj (RA2311056010026).
Generates fresh custom figures dynamically (not reusing repo figures).
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


# ============================================================================
# Step 1: Generate Fresh Custom Figures Dynamically
# ============================================================================

assets_dir = PROJECT_ROOT / "docs" / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)

# Figure 1: Conceptual System Workflow Diagram
fig1_path = assets_dir / "custom_system_workflow.png"
fig, ax = plt.subplots(figsize=(8, 2.3), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

# Draw 4 styled boxes
box_props = dict(boxstyle="round,pad=0.4", facecolor="#e8f4f8", edgecolor="#1f77b4", lw=1.5)
box_props2 = dict(boxstyle="round,pad=0.4", facecolor="#eafaf1", edgecolor="#2ca02c", lw=1.5)
box_props3 = dict(boxstyle="round,pad=0.4", facecolor="#fef5e7", edgecolor="#f39c12", lw=1.5)
box_props4 = dict(boxstyle="round,pad=0.4", facecolor="#f4ecf7", edgecolor="#8e44ad", lw=1.5)

ax.text(1.2, 1.5, "1. Graph Ingestion\n& Leak-Free Split\n(80% Train, 20% Test)", ha="center", va="center", fontsize=8.5, weight="bold", bbox=box_props)
ax.text(3.7, 1.5, "2. Representation\n& Embedding\n(Node2Vec, GNNs)", ha="center", va="center", fontsize=8.5, weight="bold", bbox=box_props2)
ax.text(6.3, 1.5, "3. Link Prediction\n& Scoring Engine\n(Hadamard, Decoder)", ha="center", va="center", fontsize=8.5, weight="bold", bbox=box_props3)
ax.text(8.8, 1.5, "4. Interactive\nStreamlit Dashboard\n(Plotly Ego-Zoom)", ha="center", va="center", fontsize=8.5, weight="bold", bbox=box_props4)

# Arrows
arrow_props = dict(arrowstyle="->", lw=2, color="#34495e")
ax.annotate("", xy=(2.4, 1.5), xytext=(2.0, 1.5), arrowprops=arrow_props)
ax.annotate("", xy=(5.0, 1.5), xytext=(4.6, 1.5), arrowprops=arrow_props)
ax.annotate("", xy=(7.6, 1.5), xytext=(7.2, 1.5), arrowprops=arrow_props)

plt.tight_layout()
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close()


# Figure 2: The Hub Bias vs. Community Homophily Phenomenon
fig2_path = assets_dir / "custom_hub_bias_concept.png"
fig, ax = plt.subplots(figsize=(8, 2.4), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

# Circle for Faction A
circle_a = patches.Ellipse((2.8, 2.0), 4.2, 3.2, angle=0, facecolor="#ebf5fb", edgecolor="#2980b9", lw=1.5, ls="--")
ax.add_patch(circle_a)
ax.text(2.8, 3.3, "Faction 1: Mr. Hi's Community", ha="center", fontsize=8, weight="bold", color="#1b4f72")

# Circle for Faction B
circle_b = patches.Ellipse((7.2, 2.0), 4.2, 3.2, angle=0, facecolor="#fef9e7", edgecolor="#d4ac0d", lw=1.5, ls="--")
ax.add_patch(circle_b)
ax.text(7.2, 3.3, "Faction 2: Opposing Faction", ha="center", fontsize=8, weight="bold", color="#7d6608")

# Nodes
ax.plot(1.8, 2.0, "o", color="#2980b9", markersize=14)
ax.text(1.8, 2.0, "U0", ha="center", va="center", color="white", weight="bold", fontsize=7.5)

ax.plot(3.8, 2.0, "o", color="#27ae60", markersize=14)
ax.text(3.8, 2.0, "U16", ha="center", va="center", color="white", weight="bold", fontsize=7.5)

ax.plot(7.5, 2.0, "o", color="#e74c3c", markersize=18)
ax.text(7.5, 2.0, "U33", ha="center", va="center", color="white", weight="bold", fontsize=7.5)

# Intermediate mutuals
ax.plot(5.0, 2.6, "o", color="#7f8c8d", markersize=8)
ax.plot(5.0, 1.4, "o", color="#7f8c8d", markersize=8)
ax.text(5.0, 2.9, "Mutual Peripheral Friends", ha="center", fontsize=6.5, color="#555")

# Lines
ax.plot([1.8, 5.0], [2.0, 2.6], "k-", lw=0.8, alpha=0.5)
ax.plot([1.8, 5.0], [2.0, 1.4], "k-", lw=0.8, alpha=0.5)
ax.plot([7.5, 5.0], [2.0, 2.6], "k-", lw=0.8, alpha=0.5)
ax.plot([7.5, 5.0], [2.0, 1.4], "k-", lw=0.8, alpha=0.5)

# Highlight edges
ax.annotate("Heuristics Pick U33\n(Hub Bias: 4 mutuals)", xy=(6.5, 2.0), xytext=(6.5, 0.7),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
            ha="center", fontsize=7, color="#c0392b", weight="bold")

ax.annotate("AI (GCN / Node2Vec) Picks U16\n(True Faction Homophily)", xy=(2.8, 2.0), xytext=(2.8, 0.7),
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2),
            ha="center", fontsize=7, color="#27ae60", weight="bold")

plt.tight_layout()
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()


# Figure 3: Benchmark Performance Chart
fig3_path = assets_dir / "custom_benchmark_summary.png"
fig, ax = plt.subplots(figsize=(8, 2.5), dpi=300)

algorithms = [
    "Common Neighbors",
    "Jaccard Sim.",
    "Pref. Attach.",
    "Node2Vec Cosine",
    "Supervised LR",
    "GraphSAGE",
    "GCN Predictor",
]
roc_auc = [0.6333, 0.6067, 0.6733, 0.7422, 0.7822, 0.7333, 0.8311]
avg_prec = [0.5784, 0.5445, 0.6794, 0.7033, 0.7676, 0.6775, 0.8769]

x = np.arange(len(algorithms))
width = 0.35

rects1 = ax.bar(x - width/2, roc_auc, width, label="ROC-AUC", color="#2980b9", edgecolor="black", lw=0.5)
rects2 = ax.bar(x + width/2, avg_prec, width, label="Average Precision", color="#27ae60", edgecolor="black", lw=0.5)

ax.set_ylabel("Score", fontsize=8, weight="bold")
ax.set_title("Comparative Model Performance on Held-Out Test Set (Karate Club)", fontsize=9, weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(algorithms, fontsize=7.5, rotation=15, ha="right")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", fontsize=7.5)
ax.grid(axis="y", linestyle="--", alpha=0.4)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6.5)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6.5)

plt.tight_layout()
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================================
# Step 2: Numbered Canvas for Running Header/Footer
# ============================================================================

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Social Network Recommendation Engine — Project Technical Report")
            self.drawRightString(558, 755, "N S Balaji & Yuvraj")
            self.setStrokeColor(colors.HexColor("#dddddd"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential & Academic — Machine Learning for Graph Representation & Link Prediction")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


# ============================================================================
# Step 3: Build the 3-Page Document
# ============================================================================

pdf_path = PROJECT_ROOT / "docs" / "Social_Network_Recommendation_Report.pdf"
doc = SimpleDocTemplate(
    str(pdf_path),
    pagesize=letter,
    leftMargin=50,
    rightMargin=50,
    topMargin=45,
    bottomMargin=50,
)

styles = getSampleStyleSheet()

# Custom styles
style_title = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    textColor=colors.HexColor("#1b2631"),
    alignment=1,  # Center
)

style_authors = ParagraphStyle(
    "DocAuthors",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10.5,
    leading=14,
    textColor=colors.HexColor("#2c3e50"),
    alignment=1,  # Center
)

style_h1 = ParagraphStyle(
    "SectionH1",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    textColor=colors.HexColor("#1a5276"),
    spaceBefore=4,
    spaceAfter=3,
)

style_body = ParagraphStyle(
    "BodyTextCustom",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=11.5,
    textColor=colors.HexColor("#2c3e50"),
    spaceAfter=4,
)

style_bullet = ParagraphStyle(
    "BulletCustom",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=11.5,
    textColor=colors.HexColor("#2c3e50"),
    leftIndent=12,
    spaceAfter=2,
)

story = []

# ============================================================================
# PAGE 1
# ============================================================================

story.append(Paragraph("Social Network Recommendation Engine", style_title))
story.append(Spacer(1, 2))
story.append(Paragraph("Graph Representation Learning, Topological Heuristics & Deep GNNs", ParagraphStyle("Sub", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=colors.HexColor("#2980b9"), alignment=1)))
story.append(Spacer(1, 4))
story.append(Paragraph("<b>Authors:</b> <b>N S Balaji</b> (Reg. No: <b>RA2311056010012</b>) &nbsp;|&nbsp; <b>Yuvraj</b> (Reg. No: <b>RA2311056010026</b>)", style_authors))
story.append(Spacer(1, 2))
story.append(Paragraph("<i>Department of Computer Science & Engineering &nbsp;•&nbsp; Academic Project Report</i>", ParagraphStyle("Sub2", fontName="Helvetica-Oblique", fontSize=8, leading=10, textColor=colors.HexColor("#7f8c8d"), alignment=1)))
story.append(Spacer(1, 6))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1f77b4"), spaceAfter=6, spaceBefore=0))

story.append(Paragraph("1. Executive Summary & Problem Formulation", style_h1))
story.append(Paragraph(
    "In modern social media platforms (Instagram, LinkedIn, Facebook), recommending new connections is fundamentally framed as <b>Link Prediction</b> on an undirected graph <i>G = (V, E)</i>. Given observed friendships <i>E</i> among users <i>V</i>, the system predicts the likelihood <i>P(Link | u, v)</i> of future friendships for unobserved pairs. Rather than treating users as isolated database rows, Graph ML leverages the <b>relational topology</b> and <b>latent community homophily</b> to suggest meaningful, high-trust connections.",
    style_body
))

story.append(Paragraph("2. System Architecture & Workflow", style_h1))
story.append(Paragraph(
    "The project implements a complete, leak-free pipeline across four developmental tiers: raw graph ingestion, bridge-preserving edge partitioning, 7 multi-paradigm recommendation models, and a responsive Streamlit visual dashboard.",
    style_body
))
story.append(Image(str(fig1_path), width=6.8*inch, height=1.9*inch))
story.append(Spacer(1, 4))

story.append(Paragraph("3. Datasets Used & Structural Provenance", style_h1))
story.append(Paragraph(
    "The models are empirically benchmarked across two distinct graph scales:",
    style_body
))
story.append(Paragraph(
    "• <b>Zachary's Karate Club (Zachary, 1977):</b> A classical human fission network of <b>34 members</b> and <b>78 friendships</b> with an observed political feud between the instructor ('Mr. Hi', Node 0) and administrator ('John A.', Node 33). Serves as ground truth for community detection.",
    style_bullet
))
story.append(Paragraph(
    "• <b>Stanford SNAP Facebook Social Circles (McAuley & Leskovec, 2012):</b> Anonymized real-world Facebook data tested at two scales: <b>Facebook Ego-0</b> (a dense local circle of <b>348 nodes, 2,866 edges</b>, density = 4.7%) and <b>Facebook Full</b> (a global web of <b>4,039 nodes, 88,234 edges</b>, density = 1.08%).",
    style_bullet
))

# Small dataset table
table_data1 = [
    ["Dataset Name", "Nodes (|V|)", "Edges (|E|)", "Density", "Avg Degree", "Clustering (C)"],
    ["Zachary's Karate Club", "34", "78", "0.1390", "4.59", "0.5706"],
    ["SNAP Facebook Ego-0", "348", "2,866", "0.0474", "16.47", "0.6554"],
    ["SNAP Facebook Full", "4,039", "88,234", "0.0108", "43.69", "0.6055"],
]
t1 = Table(table_data1, colWidths=[1.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 1.0*inch, 1.2*inch])
t1.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b4f72")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9f9"), colors.white]),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story.append(t1)

# End of Page 1
story.append(PageBreak())

# ============================================================================
# PAGE 2
# ============================================================================

story.append(Paragraph("4. Algorithmic Paradigms & Mathematical Mechanics", style_h1))
story.append(Paragraph(
    "Seven distinct link prediction algorithms spanning four methodological paradigms were developed from scratch:",
    style_body
))

story.append(Paragraph(
    "<b>A. Topological Heuristics (Rules of Thumb):</b><br/>"
    "• <i>Common Neighbors (CN):</i> Scores connection potential purely via shared friends: <i>S(u, v) = |N(u) ∩ N(v)|</i>.<br/>"
    "• <i>Jaccard Similarity:</i> Normalizes overlap by combined footprint: <i>S(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|</i>, penalizing celebrity hubs.<br/>"
    "• <i>Preferential Attachment (PA):</i> Models rich-get-richer growth: <i>S(u, v) = deg(u) × deg(v)</i>, capturing global popularity.",
    style_bullet
))

story.append(Paragraph(
    "<b>B. Graph Representation Learning (Node2Vec):</b><br/>"
    "Treats biased 2nd-order random walks as sentences and optimizes continuous 16-dimensional coordinate vectors <b>z</b>_u via Skip-gram with negative sampling. Controlled by return parameter <i>p</i> (local homophily) and in-out parameter <i>q</i> (structural exploration). Scored via cosine angle: <i>S(u, v) = (<b>z</b>_u · <b>z</b>_v) / (||<b>z</b>_u|| ||<b>z</b>_v||)</i>.",
    style_bullet
))

story.append(Paragraph(
    "<b>C. Supervised Machine Learning (Hadamard + Logistic Regression):</b><br/>"
    "Constructs element-wise symmetric edge features: <b>x</b>_(u, v) = <b>z</b>_u ⊙ <b>z</b>_v. Because multiplication is commutative (<i>z_(u,i) · z_(v,i) ≡ z_(v,i) · z_(u,i)</i>), order invariance is guaranteed for undirected social ties. Trains an L2-regularized classifier outputting calibrated probabilities: <i>P(Link) = σ(<b>w</b>^T <b>x</b>_(u, v) + b)</i>.",
    style_bullet
))

story.append(Paragraph(
    "<b>D. Deep Graph Neural Networks (GCN & GraphSAGE):</b><br/>"
    "• <i>Graph Convolutional Network (GCN):</i> Performs spectral message passing: <b>H</b>^(l+1) = σ(<b>D</b>~^(-1/2) <b>A</b>~ <b>D</b>~^(-1/2) <b>H</b>^(l) <b>W</b>^(l)) with self-loops and democratic degree normalization.<br/>"
    "• <i>GraphSAGE:</i> Inductively aggregates neighborhood mean vectors and concatenates with node state. Decodes links end-to-end via inner product: <i>y^_uv = σ(<b>z</b>_u^T <b>z</b>_v)</i> minimizing Binary Cross-Entropy loss.",
    style_bullet
))
story.append(Spacer(1, 4))

story.append(Paragraph("5. The Hub Bias Problem & Community Homophily", style_h1))
story.append(Paragraph(
    "A fundamental discovery in this project is the vulnerability of counting heuristics to <b>Hub Bias</b>: popular individuals (User 33) accumulate mutual connections purely through volume, misleading Common Neighbors into recommending rival leaders. Deep GNNs and Node2Vec eliminate hub bias by capturing multi-hop community homophily (prioritizing true faction peer User 16).",
    style_body
))
story.append(Image(str(fig2_path), width=6.8*inch, height=2.0*inch))
story.append(Spacer(1, 4))

story.append(Paragraph("6. Strict Leak-Free Data Partitioning Protocol", style_h1))
story.append(Paragraph(
    "To prevent data leakage, 20% test edges are withheld from the training graph <i>G_train</i>. We enforce a <b>bridge-preserving check</b> (<i>nx.is_connected</i>) ensuring no nodes become isolated. Node2Vec random walks and GCN message-passing edges (<i>edge_index</i>) are constructed <b>exclusively on G_train</b>, while negative non-edges are sampled disjointly with zero train/test overlap.",
    style_body
))

# End of Page 2
story.append(PageBreak())

# ============================================================================
# PAGE 3
# ============================================================================

story.append(Paragraph("7. Evaluation Metrics & Empirical Findings", style_h1))
story.append(Paragraph(
    "Models are evaluated across global edge discrimination (<b>ROC-AUC</b> and <b>Average Precision / PR-AUC</b>) and user-centric ranking quality (<b>Precision@5</b> and <b>Recall@5</b>). Below are the empirical benchmark results on held-out test splits:",
    style_body
))

# Benchmark Table
table_data2 = [
    ["Model Name", "Paradigm Category", "ROC-AUC (Karate)", "AP (Karate)", "ROC-AUC (Ego-0)", "AP (Ego-0)"],
    ["Common Neighbors", "Topological Heuristic", "0.6333", "0.5784", "0.9368", "0.9298"],
    ["Jaccard Similarity", "Topological Heuristic", "0.6067", "0.5445", "0.8989", "0.8809"],
    ["Preferential Attachment", "Topological Heuristic", "0.6733", "0.6794", "0.6346", "0.6121"],
    ["Node2Vec Cosine", "AI Embedding", "0.7422", "0.7033", "0.8780", "0.8864"],
    ["Supervised LogReg", "Supervised ML", "0.7822", "0.7676", "0.8802", "0.8879"],
    ["GraphSAGE Predictor", "Deep GNN", "0.7333", "0.6775", "—", "—"],
    ["GCN Link Predictor", "Deep GNN", "0.8311", "0.8769", "—", "—"],
]
t2 = Table(table_data2, colWidths=[1.7*inch, 1.4*inch, 1.0*inch, 0.9*inch, 1.0*inch, 0.8*inch])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b4f72")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9f9"), colors.white]),
    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#d5f5e3")),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story.append(t2)
story.append(Spacer(1, 4))

story.append(Image(str(fig3_path), width=6.8*inch, height=2.1*inch))
story.append(Spacer(1, 4))

story.append(Paragraph("8. Key Scientific Insights & The Ego Inversion", style_h1))
story.append(Paragraph(
    "• <b>GCN SOTA Performance:</b> The 2-layer GCN Link Predictor achieved the highest score on Zachary's Karate Club (<b>ROC-AUC: 0.8311, AP: 0.8769</b>), validating end-to-end message passing over shallow two-stage embeddings.<br/>"
    "• <b>The Ego Inversion:</b> On dense personal ego networks (Facebook Ego-0, <i>C = 0.655</i>), <b>Common Neighbors dominates (ROC-AUC: 0.9368)</b> because triadic closure is the primary mechanism of clique formation. On global networks, representation learning is essential to discover multi-hop community structure.",
    style_body
))

story.append(Paragraph("9. Interactive Streamlit Dashboard & Production Testing", style_h1))
story.append(Paragraph(
    "The accompanying Streamlit application (<i>app.py</i>) serves all models with real-time Plotly graph visualization. To eliminate browser freezing on large graphs (88,000+ edges), an automated <b>2-hop ego-neighborhood extraction</b> renders subgraphs in under 0.2s. The codebase is validated with <b>67 automated unit tests</b> passing with zero errors.",
    style_body
))

story.append(Spacer(1, 4))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#95a5a6"), spaceAfter=4, spaceBefore=2))
story.append(Paragraph(
    "<b>Project Deliverables & Repository:</b> Source code, trained checkpoints, unit tests, and interactive dashboard are live on GitHub: <i>https://github.com/NS145/Social-Network-Recommendation</i>.",
    ParagraphStyle("FooterNote", fontName="Helvetica-Oblique", fontSize=7.5, leading=9.5, textColor=colors.HexColor("#555555"), alignment=1)
))

# Build Document
doc.build(story, canvasmaker=NumberedCanvas)
print("PDF generated successfully at:", pdf_path)

