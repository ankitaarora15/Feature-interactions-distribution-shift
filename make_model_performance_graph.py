from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path.cwd()
df = pd.read_csv(DATA_DIR / "consolidated_shift_results.csv")
order = ["Baseline", "Age", "Hours", "Education", "Age+Hours", "Age+Education", "Hours+Education", "Age+Hours+Education"]
models = ["Logistic Regression", "Decision Tree", "Random Forest"]
fig, ax = plt.subplots(figsize=(12, 7))
y = list(range(len(order))); h = 0.23
for j, model in enumerate(models):
    subset = df[df["model"] == model].set_index("experiment")
    values = [subset.loc[name, "balanced_accuracy"] for name in order]
    ax.barh([v + (j - 1) * h for v in y], values, height=h, label=model)
ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis()
ax.set_xlabel("Balanced accuracy"); ax.set_ylabel("Evaluation condition")
ax.set_title("Balanced Accuracy Under Class-Preserving Subgroup Reweighting (×3)")
ax.grid(axis="x", alpha=0.25); ax.legend(); fig.tight_layout()
fig.savefig(DATA_DIR / "model_performance_graph.png", dpi=300, bbox_inches="tight")
plt.show()
