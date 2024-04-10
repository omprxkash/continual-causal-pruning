"""
Code contribution visualization.

Generates a breakdown of lines of code per module and a file-size
heatmap for the continual-causal-pruning project.

Run from repo root:
    python results/plot_contributions.py
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Count non-empty, non-comment lines per file
# ---------------------------------------------------------------------------

def count_loc(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return sum(
            1 for l in lines
            if l.strip() and not l.strip().startswith("#")
        )
    except Exception:
        return 0


FILES = {
    # src/
    "preprocess.py":       ROOT / "src" / "preprocess.py",
    "train.py":            ROOT / "src" / "train.py",
    "evaluate.py":         ROOT / "src" / "evaluate.py",
    "models/base_model":   ROOT / "src" / "models" / "base_model.py",
    "models/ewc":          ROOT / "src" / "models" / "ewc.py",
    "models/gem":          ROOT / "src" / "models" / "gem.py",
    "models/packnet":      ROOT / "src" / "models" / "packnet.py",
    "models/causal":       ROOT / "src" / "models" / "causal_pruning.py",
    # results/
    "generate_results":    ROOT / "results" / "generate_results.py",
    "plot_contributions":  ROOT / "results" / "plot_contributions.py",
}

NOTEBOOK_FILES = {
    f"nb/{p.stem[:18]}": p
    for p in sorted((ROOT / "notebooks").glob("*.ipynb"))
}

COLORS = {
    "preprocess.py":      "#3498db",
    "train.py":           "#2ecc71",
    "evaluate.py":        "#f39c12",
    "models/base_model":  "#9b59b6",
    "models/ewc":         "#e74c3c",
    "models/gem":         "#1abc9c",
    "models/packnet":     "#e67e22",
    "models/causal":      "#8e44ad",
    "generate_results":   "#95a5a6",
    "plot_contributions": "#7f8c8d",
}

NB_COLOR = "#2980b9"


# ---------------------------------------------------------------------------
# Build data
# ---------------------------------------------------------------------------

src_names = list(FILES.keys())
src_locs  = [count_loc(FILES[n]) for n in src_names]

nb_names = list(NOTEBOOK_FILES.keys())
nb_locs  = [count_loc(p) for p in NOTEBOOK_FILES.values()]

all_names = src_names + nb_names
all_locs  = src_locs  + nb_locs
all_colors = [COLORS.get(n, NB_COLOR) for n in all_names]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                         gridspec_kw={"width_ratios": [2, 1]})
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")

# ---- Panel 1: Horizontal bar chart (LOC per file) ----------------------
sorted_idx = np.argsort(all_locs)[::-1]
s_names  = [all_names[i]  for i in sorted_idx]
s_locs   = [all_locs[i]   for i in sorted_idx]
s_colors = [all_colors[i] for i in sorted_idx]

y = np.arange(len(s_names))
bars = axes[0].barh(y, s_locs, color=s_colors, alpha=0.85, height=0.7,
                    edgecolor="none")

# Labels
for bar, loc in zip(bars, s_locs):
    axes[0].text(loc + 2, bar.get_y() + bar.get_height() / 2,
                 str(loc), va="center", color="white", fontsize=9)

axes[0].set_yticks(y)
axes[0].set_yticklabels(s_names, color="white", fontsize=9)
axes[0].set_xlabel("Lines of code (non-empty, non-comment)",
                   color="white", fontsize=10)
axes[0].set_title("Code Contribution by File",
                  color="white", fontsize=12, fontweight="bold", pad=10)
axes[0].tick_params(colors="white")
axes[0].spines[:].set_color("#30363d")
axes[0].xaxis.label.set_color("white")
axes[0].set_xlim(0, max(s_locs) * 1.15)

# ---- Panel 2: Pie chart (module groups) ---------------------------------
groups = {
    "Models\n(ewc/gem/packnet/causal)": sum(
        count_loc(ROOT / "src" / "models" / f)
        for f in ["ewc.py", "gem.py", "packnet.py", "causal_pruning.py", "base_model.py"]
    ),
    "Training\n& Evaluation": sum(
        count_loc(ROOT / "src" / f) for f in ["train.py", "evaluate.py", "preprocess.py"]
    ),
    "Notebooks\n(01–07)": sum(nb_locs),
    "Results\n& Plotting": sum(
        count_loc(ROOT / "results" / f) for f in ["generate_results.py", "plot_contributions.py"]
    ),
}

pie_colors  = ["#8e44ad", "#2ecc71", "#3498db", "#95a5a6"]
wedge_props = {"linewidth": 1.5, "edgecolor": "#0d1117"}

wedges, texts, autotexts = axes[1].pie(
    groups.values(),
    labels=groups.keys(),
    colors=pie_colors,
    autopct="%1.0f%%",
    wedgeprops=wedge_props,
    startangle=140,
    pctdistance=0.75,
    textprops={"color": "white", "fontsize": 9},
)
for at in autotexts:
    at.set_color("white")
    at.set_fontsize(10)
    at.set_fontweight("bold")

axes[1].set_title("Module Distribution",
                  color="white", fontsize=12, fontweight="bold", pad=10)

total_loc = sum(all_locs)
axes[1].text(0, -1.45, f"Total: {total_loc:,} LOC",
             ha="center", color="#8b949e", fontsize=10)

plt.suptitle("continual-causal-pruning — Code Contribution",
             color="white", fontsize=14, fontweight="bold", y=1.01)

plt.tight_layout()
out = ROOT / "results" / "code_contribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
