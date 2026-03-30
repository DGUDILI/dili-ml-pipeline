"""DGUDILI Algorithm Flowchart Generator."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

fig, ax = plt.subplots(figsize=(14, 22))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor("#F8F9FA")

C_DATA  = "#4A90D9"
C_PROC  = "#5BAD6F"
C_MODEL = "#E8A838"
C_ATTN  = "#C0392B"
C_OUT   = "#8E44AD"
C_ARROW = "#555555"


def box(ax, x, y, w, h, text, color, fontsize=9, text_color="white"):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.1",
                       facecolor=color, edgecolor="white", linewidth=1.5, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=4, multialignment="center")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.8, mutation_scale=14),
                zorder=2)


def line(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=C_ARROW, lw=1.5, zorder=2)


# Title
ax.text(7, 21.3, "DGUDILI Algorithm Flowchart",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#2C3E50")

# 1. Input
box(ax, 7, 20.3, 4.5, 0.7, "Dataset.csv\n(SMILES | Label | ref)  n=1,850", C_DATA)

line(ax, 7, 19.95, 7, 19.75)
line(ax, 7, 19.75, 3.5, 19.75)
line(ax, 7, 19.75, 10.5, 19.75)
arrow(ax, 3.5, 19.75, 3.5, 19.2)
arrow(ax, 10.5, 19.75, 10.5, 19.2)

# 2. Feature extraction
box(ax, 3.5, 18.8, 4.2, 0.7,
    "Feature.py\n(iFeatureOmegaCLI)", C_PROC)
box(ax, 10.5, 18.8, 4.2, 0.7,
    "chemberta_encoder.py\n(seyonec/ChemBERTa-zinc-base-v1)", C_PROC, fontsize=8)

arrow(ax, 3.5, 18.45, 3.5, 17.85)
arrow(ax, 10.5, 18.45, 10.5, 17.85)

# 3. Output files
box(ax, 3.5, 17.55, 4.2, 0.55,
    "dataset_features.csv  (n x 425 FP)", C_DATA, fontsize=8.5)
box(ax, 10.5, 17.55, 4.2, 0.55,
    "chemberta_embeddings.npy  (n x 768)", C_DATA, fontsize=8.5)

arrow(ax, 3.5, 17.27, 3.5, 16.7)
box(ax, 3.5, 16.35, 4.6, 0.65,
    "FP Group Split  (get_fp_groups)\n"
    "const(173) | pc(6) | maccs(167) | estate(79)",
    C_PROC, fontsize=7.8)

arrow(ax, 3.5, 16.02, 3.5, 15.5)
line(ax, 10.5, 17.27, 10.5, 15.5)
line(ax, 3.5, 15.5, 10.5, 15.5)
line(ax, 7, 15.5, 7, 15.22)
arrow(ax, 7, 15.22, 7, 14.82)

# 4. DGUDILIModel outer box
model_box = FancyBboxPatch((1.5, 9.8), 11, 4.9,
                           boxstyle="round,pad=0.15",
                           facecolor="#FFF8E7", edgecolor=C_MODEL,
                           linewidth=2, zorder=1)
ax.add_patch(model_box)
ax.text(7, 14.55, "DGUDILIModel  (PyTorch nn.Module)",
        ha="center", va="center", fontsize=10, fontweight="bold", color=C_MODEL, zorder=4)

# FP projections
box(ax, 3.5, 13.8, 4.2, 0.9,
    "FP Projections  (ModuleList x4)\n"
    "Linear(173,64)+LayerNorm+ReLU\n"
    "Linear(  6,64)+LayerNorm+ReLU\n"
    "Linear(167,64)+LayerNorm+ReLU\n"
    "Linear( 79,64)+LayerNorm+ReLU",
    C_MODEL, fontsize=7.3)

# CB projection
box(ax, 10.5, 13.8, 4.2, 0.9,
    "CB Projection\n"
    "Linear(768, 64)\n"
    "+LayerNorm+ReLU\n"
    "-> (batch, 1, 64)",
    C_MODEL, fontsize=7.8)

ax.text(3.5, 13.2, "-> stack -> (batch, 4, 64)",
        ha="center", va="center", fontsize=7.5, color="#555", zorder=4)

arrow(ax, 3.5, 13.25, 3.5, 12.75)
arrow(ax, 10.5, 13.25, 10.5, 12.75)

# Mode A / B
box(ax, 3.5, 12.4, 4.2, 0.6,
    "Mode A:  Q=FP(4 tokens)  K=V=CB(1 token)\n"
    "-> attn_out (batch,4,64) -> flatten -> Linear(256,16)",
    C_ATTN, fontsize=7.3)
box(ax, 10.5, 12.4, 4.2, 0.6,
    "Mode B:  Q=CB(1 token)  K=V=FP(4 tokens)\n"
    "-> attn_out (batch,1,64) -> squeeze -> Linear(64,16)",
    C_ATTN, fontsize=7.3)

ax.text(7, 11.9, "MultiheadAttention  (d_model=64, n_heads=4)",
        ha="center", va="center", fontsize=8, color=C_ATTN, fontweight="bold", zorder=4)

# Residual + 16-dim
box(ax, 7, 11.3, 5.0, 0.55,
    "Residual + LayerNorm  ->  16-dim feature  (batch, 16)",
    C_MODEL, fontsize=8)

# Training head
box(ax, 7, 10.35, 4.5, 0.75,
    "[Train] Linear(16->1) logit\n"
    "BCEWithLogitsLoss (pos_weight balanced)\n"
    "Adam  lr=1e-3  WD=1e-4  CosineAnnealing  epochs=80",
    "#7F8C8D", fontsize=7.3)

arrow(ax, 7, 11.02, 7, 10.73)
arrow(ax, 7, 9.8, 7, 9.3)

# 5. Inference
box(ax, 7, 9.0, 5.0, 0.55,
    "extract_features()  ->  (n, 16) numpy array",
    C_PROC, fontsize=8.5)

arrow(ax, 7, 8.72, 7, 8.22)

box(ax, 7, 7.9, 4.5, 0.55,
    "StandardScaler  +  LogisticRegression\n"
    "(C=1.0, max_iter=1000, class_weight='balanced')",
    C_PROC, fontsize=8)

arrow(ax, 7, 7.62, 7, 7.1)

# 6. Protocol split
box(ax, 7, 6.8, 3.5, 0.55, "Evaluation Protocol", "#2C3E50", fontsize=9)

line(ax, 7, 6.52, 7, 6.35)
line(ax, 7, 6.35, 3.5, 6.35)
line(ax, 7, 6.35, 10.5, 6.35)
arrow(ax, 3.5, 6.35, 3.5, 5.85)
arrow(ax, 10.5, 6.35, 10.5, 5.85)

box(ax, 3.5, 5.55, 4.2, 0.55,
    "env1: External Validation\nNCTR+Greene+Xu+Liew train\n-> DILIrank test",
    C_OUT, fontsize=7.8)
box(ax, 10.5, 5.55, 4.2, 0.55,
    "env2: 10-Fold Stratified CV\nFull dataset\n(n=1,850)",
    C_OUT, fontsize=7.8)

arrow(ax, 3.5, 5.27, 3.5, 4.72)
arrow(ax, 10.5, 5.27, 10.5, 4.72)

# 7. Results
box(ax, 3.5, 4.4, 4.5, 0.65,
    "Mode A:  AUC=0.8677  MCC=0.5565\n"
    "Mode B:  AUC=0.9176  MCC=0.7467",
    C_OUT, fontsize=8)
box(ax, 10.5, 4.4, 4.0, 0.65,
    "AUC_mean / MCC_mean\n+/- std across 10 folds",
    C_OUT, fontsize=8)

line(ax, 3.5, 4.07, 3.5, 3.55)
line(ax, 3.5, 3.55, 7, 3.55)
arrow(ax, 7, 3.55, 7, 3.12)

# 8. Target
box(ax, 7, 2.8, 7.5, 0.65,
    "Target: AUC >= 0.930  MCC >= 0.480\n"
    "Result: Mode B  AUC=0.9176 (close)  MCC=0.7467 (exceeded)",
    "#27AE60", fontsize=8.5)

# Legend
legend_items = [
    (C_DATA,  "Data file"),
    (C_PROC,  "Processing"),
    (C_MODEL, "Model layer"),
    (C_ATTN,  "Cross-Attention"),
    (C_OUT,   "Evaluation"),
]
for i, (c, label) in enumerate(legend_items):
    rx, ry = 1.0 + i * 2.5, 1.6
    p = FancyBboxPatch((rx, ry - 0.18), 0.35, 0.35,
                       boxstyle="round,pad=0.05",
                       facecolor=c, edgecolor="white", lw=1, zorder=3)
    ax.add_patch(p)
    ax.text(rx + 0.5, ry, label, va="center", fontsize=7.5, color="#2C3E50", zorder=4)

plt.tight_layout(pad=0.5)
out = "c:/dili-ml-pipeline/docs/dgudili_flowchart.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
