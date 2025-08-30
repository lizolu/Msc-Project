import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load dataset ===
path = "Dataset\Women's clothing review results.csv"   # <-- adjust path if needed
df = pd.read_csv(path)

# === Identify relevant columns (adjust names if different in your CSV) ===
hyb_col = "Hybrid Category"      # column containing category labels
rec_col = "Recommended IND"      # column containing 0/1 recommendation

# === Map Recommended IND -> sentiment ===
#   1 = Positive, 0 = Negative
df["Sentiment"] = np.where(df[rec_col] == 1, "Positive", "Negative")

# === Pivot table of counts ===
counts = (
    df.groupby([hyb_col, "Sentiment"])
      .size()
      .unstack(fill_value=0)
      .sort_index()
)

# Ensure Positive/Negative order
ordered_cols = [c for c in ["Positive", "Negative"] if c in counts.columns]
counts = counts[ordered_cols]

# Proportions for 100% stacked bar
props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

# === Visualization 1: Stacked bar (Counts) ===
fig1, ax1 = plt.subplots(figsize=(11, 6))
x = np.arange(len(counts.index))
bottom = np.zeros(len(counts))

for col in counts.columns:
    ax1.bar(x, counts[col].values, bottom=bottom, label=col)
    bottom += counts[col].values

ax1.set_xticks(x)
ax1.set_xticklabels(counts.index, rotation=30, ha="right")
ax1.set_ylabel("Number of Reviews")
ax1.set_title("Sentiment by Hybrid Category (Counts)")
ax1.legend(title="Sentiment", loc="best")

# Add total values above bars
totals = counts.sum(axis=1).values
for i, total in enumerate(totals):
    ax1.text(i, total, str(int(total)), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()

# === Visualization 2: 100% Stacked bar (Proportions) ===
fig2, ax2 = plt.subplots(figsize=(11, 6))
x = np.arange(len(props.index))
bottom = np.zeros(len(props))

for col in props.columns:
    ax2.bar(x, props[col].values, bottom=bottom, label=col)
    bottom += props[col].values

ax2.set_xticks(x)
ax2.set_xticklabels(props.index, rotation=30, ha="right")
ax2.set_ylabel("Proportion of Reviews")
ax2.set_title("Sentiment by Hybrid Category (Proportions)")
ax2.legend(title="Sentiment", loc="best")

# Annotate percentages inside bars
for i in range(len(props.index)):
    cum = 0.0
    for col in props.columns:
        val = props.iloc[i][col]
        if val > 0.04:  # only label if at least 4% of bar
            ax2.text(i, cum + val/2, f"{val*100:.0f}%", ha="center", va="center", fontsize=9)
        cum += val

plt.tight_layout()
plt.show()

# === Save counts table to CSV (optional) ===
counts.to_csv("hybrid_category_sentiment_counts.csv")
print("Saved counts table to 'hybrid_category_sentiment_counts.csv'")
