import xlsxwriter 
import pandas as pd
# ===== Group reviews by sentiment within each Hybrid Category =====
df= pd.read_csv("Dataset\Women's clothing review results.csv",index_col=0)
# 0) Ensure there's a usable Sentiment column
#    If your dataset already has one (e.g., "Sentiment" with values like Positive/Neutral/Negative),
#    this will use it. Otherwise, we derive it from a numeric "Rating" if present.
if "Sentiment" not in df.columns:
    if "Rating" in df.columns:
        # Map rating to sentiment: 1–2 = Negative, 3 = Neutral, 4–5 = Positive
        df["Sentiment"] = pd.cut(
            df["Rating"],
            bins=[0, 2, 3, 5],
            labels=["Negative", "Neutral", "Positive"],
            include_lowest=True,
            right=True
        ).astype(str)
    else:
        raise ValueError(
            "No 'Sentiment' column found. Add one to the dataset or include a 'Rating' column so sentiment can be derived."
        )

# Clean up any odd capitalization/whitespace
df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.title()

# 1) Quick counts table: Hybrid Category × Sentiment
counts_table = (
    df.groupby(["Hybrid Category", "Sentiment"])
      .size()
      .unstack(fill_value=0)
      .sort_index()
)
print("\n=== Review counts per Hybrid Category × Sentiment ===")
print(counts_table)

# 2) Create a nested dict: {category: {sentiment: [reviews...]}}
category_sentiment_reviews = {
    cat: {
        sent: grp["Review Text"].tolist()
        for sent, grp in sub.groupby("Sentiment", sort=False)
    }
    for cat, sub in df.groupby("Hybrid Category", sort=True)
}

# (Optional) Print a few samples per bucket for a sanity check
pd.set_option("display.max_colwidth", 200)
for cat in sorted(df["Hybrid Category"].unique()):
    print(f"\n===== {cat} =====")
    for sent in ["Positive", "Neutral", "Negative"]:
        if sent in category_sentiment_reviews[cat] and len(category_sentiment_reviews[cat][sent]) > 0:
            print(f"\n-- {sent} (showing up to 3) --")
            for tx in category_sentiment_reviews[cat][sent][:3]:
                print(f"- {tx}")

# 3) Save to Excel:
#    - Sheet 'Counts' = pivot table of counts
#    - One sheet per Hybrid Category listing Review Text + Sentiment
#      (Sheet names truncated to 31 chars to satisfy Excel's limit.)
out_path = ".\Dataset\hybrid_category_sentiment_groups.xlsx"
with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    counts_table.to_excel(writer, sheet_name="Counts")
    for cat, sub in df.groupby("Hybrid Category", sort=True):
        sheet = cat[:31]  # Excel sheet name limit
        sub[["Review Text", "Sentiment"]].sort_values("Sentiment").to_excel(
            writer, sheet_name=sheet, index=False
        )

print(f"\nSaved grouped results to: {out_path}")
