import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.spatial.distance import pdist, squareform

# =========================
# Setup & Data Loading
# =========================
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load dataset
data_path = os.path.join("Dataset", "Womens Clothing E-Commerce Reviews.csv")
df = pd.read_csv(data_path)
df = df[:100]
df = df[~df['Review Text'].isnull()].reset_index(drop=True)

# =========================
# Category Keywords
# =========================
cluster_keywords = {
    "Style and Aesthetics": [
        "stylish", "trendy", "fashionable", "chic", "modern", "elegant", "cute",
        "beautiful", "design", "pattern", "color", "print", "classy", "look",
        "style", "flattering", "aesthetic", "unique", "versatile", "sleek"
    ],
    "Fit and Sizing": [
        "tight", "loose", "fitted", "oversized", "true to size", "small", "big",
        "fit", "sizing", "proportion", "snug", "comfortable fit", "perfect fit",
        "length", "short", "long", "size", "waist", "hips", "shoulder"
    ],
    "Fabric and Material Quality": [
        "soft", "rough", "scratchy", "smooth", "see-through", "durable",
        "delicate", "stretchy", "fabric", "material", "quality", "texture",
        "thick", "thin", "luxurious", "lightweight", "heavy", "breathable",
        "synthetic", "cotton"
    ],
    "Comfort and Wearability": [
        "comfortable", "uncomfortable", "lightweight", "breathable", "warm",
        "cool", "easy to wear", "soft", "itchy", "stretchy", "relaxed", "casual",
        "practical", "movable", "airy", "cozy", "snug", "restrictive",
        "functional", "pleasant"
    ],
    "Occasion and Use Case": [
        "work", "office", "wedding", "party", "casual", "formal", "evening",
        "daywear", "holiday", "vacation", "beach", "gym", "event", "ceremony",
        "travel", "special occasion", "weekend", "everyday", "festive", "outing"
    ],
    "Price-Value Perception": [
        "affordable", "expensive", "worth", "overpriced", "cheap", "value",
        "price", "reasonable", "budget", "deal", "cost", "investment", "bargain",
        "quality for price", "overvalued", "economical", "pricy", "low-cost",
        "steal", "costly"
    ]
}

# =========================
# Preprocessing
# =========================
def clean_text(text: str):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return words

df["Processed Review"] = df["Review Text"].apply(lambda x: clean_text(str(x)))

# =========================
# Embeddings: SBERT
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')

# Category embeddings (concatenate keywords per category into a short "description")
category_sentences = {cat: " ".join(words) for cat, words in cluster_keywords.items()}
category_embeddings = {
    cat: model.encode(desc, convert_to_tensor=True) for cat, desc in category_sentences.items()
}

# =========================
# Hybrid Assignment (keyword + semantic)
# =========================
def assign_category_hybrid(review_words, original_text, alpha=0.5):
    scores = {}
    review_embedding = model.encode(original_text, convert_to_tensor=True)
    for cat, keywords in cluster_keywords.items():
        stemmed_keywords = [stemmer.stem(w.lower()) for w in keywords]
        # Keyword overlap score (fraction of cat keywords present in review)
        keyword_score = sum(w in stemmed_keywords for w in review_words) / max(1, len(stemmed_keywords))
        # Semantic similarity via cosine (SBERT)
        semantic_score = util.pytorch_cos_sim(review_embedding, category_embeddings[cat]).item()
        # Combine
        combined_score = alpha * keyword_score + (1 - alpha) * semantic_score
        scores[cat] = combined_score
    return max(scores, key=scores.get)

df["Hybrid Category"] = df.apply(
    lambda row: assign_category_hybrid(row["Processed Review"], row["Review Text"]), axis=1
)

# =========================
# KMeans Baseline (on embeddings)
# =========================
review_embeddings = model.encode(df["Review Text"].tolist(), convert_to_tensor=False)
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df["KMeans Cluster"] = kmeans.fit_predict(review_embeddings)

# 🔹 Map numeric cluster labels to names like "Cluster 1", "Cluster 2", etc.
cluster_name_map = {i: f"Cluster {i+1}" for i in range(kmeans.n_clusters)}
df["KMeans Cluster"] = df["KMeans Cluster"].map(cluster_name_map)


# =========================
# Metrics Functions
# =========================
def dunn_index(X, labels):
    """
    Dunn index = (min inter-cluster distance) / (max intra-cluster distance).
    Higher is better.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    distances = squareform(pdist(X))
    unique_clusters = np.unique(labels)

    # Max intra-cluster distance
    intra = []
    for c in unique_clusters:
        points = np.where(labels == c)[0]
        if len(points) > 1:
            intra.append(np.max(distances[np.ix_(points, points)]))
    if len(intra) == 0:
        return np.nan
    max_intra = np.max(intra)

    # Min inter-cluster distance
    inter = []
    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            points_i = np.where(labels == unique_clusters[i])[0]
            points_j = np.where(labels == unique_clusters[j])[0]
            if len(points_i) > 0 and len(points_j) > 0:
                inter.append(np.min(distances[np.ix_(points_i, points_j)]))
    if len(inter) == 0:
        return np.nan
    min_inter = np.min(inter)

    if max_intra == 0:
        return np.nan
    return float(min_inter / max_intra)

def compute_wcss(X, labels):
    """
    Within-Cluster Sum of Squares (lower is better).
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    wcss = 0.0
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        if cluster_points.size == 0:
            continue
        centroid = cluster_points.mean(axis=0)
        wcss += ((cluster_points - centroid) ** 2).sum()
    return float(wcss)

def eval_metrics(X, labels):
    """
    Returns a dict of standard clustering metrics.
    Safeguards for degenerate cases (e.g., only 1 cluster).
    """
    X_np = np.asarray(X)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return {
            "Silhouette": np.nan,
            "DaviesBouldin": np.nan,
            "CalinskiHarabasz": np.nan,
            "Dunn": np.nan,
            "WCSS": np.nan,
        }
    return {
        "Silhouette": float(silhouette_score(X_np, labels)),
        "DaviesBouldin": float(davies_bouldin_score(X_np, labels)),
        "CalinskiHarabasz": float(calinski_harabasz_score(X_np, labels)),
        "Dunn": dunn_index(X_np, labels),
        "WCSS": compute_wcss(X_np, labels),
    }

# =========================
# Additional Approaches: Keyword-only & Semantic-only
# =========================
def assign_keyword_only(review_words):
    scores = {}
    for cat, keywords in cluster_keywords.items():
        stems = [stemmer.stem(w.lower()) for w in keywords]
        kw_score = sum(w in stems for w in review_words) / max(1, len(stems))
        scores[cat] = kw_score
    return max(scores, key=scores.get)

def assign_semantic_only(text):
    emb = model.encode(text, convert_to_tensor=True)
    scores = {
        cat: util.pytorch_cos_sim(emb, category_embeddings[cat]).item()
        for cat in cluster_keywords.keys()
    }
    return max(scores, key=scores.get)

df["Keyword_Category"]  = df["Processed Review"].apply(assign_keyword_only)
df["Semantic_Category"] = df["Review Text"].apply(assign_semantic_only)

# =========================
# Encode Labels & Evaluate
# =========================
le_kw = LabelEncoder()
le_sem = LabelEncoder()
le_h = LabelEncoder()

labels_kmeans   = df["KMeans Cluster"].values
labels_keyword  = le_kw.fit_transform(df["Keyword_Category"])
labels_semantic = le_sem.fit_transform(df["Semantic_Category"])
labels_hybrid   = le_h.fit_transform(df["Hybrid Category"])

summary_rows = []
summary_rows.append({"Approach": "KMeans"}   | eval_metrics(review_embeddings, labels_kmeans))
summary_rows.append({"Approach": "Keyword"}  | eval_metrics(review_embeddings, labels_keyword))
summary_rows.append({"Approach": "Semantic"} | eval_metrics(review_embeddings, labels_semantic))
summary_rows.append({"Approach": "Hybrid"}   | eval_metrics(review_embeddings, labels_hybrid))

summary_df = pd.DataFrame(summary_rows)[
    ["Approach", "Silhouette", "DaviesBouldin", "CalinskiHarabasz", "Dunn", "WCSS"]
].copy()

# Round for readability
summary_df[["Silhouette","DaviesBouldin","CalinskiHarabasz","Dunn","WCSS"]] = \
    summary_df[["Silhouette","DaviesBouldin","CalinskiHarabasz","Dunn","WCSS"]].round(4)

# =========================
# Save Outputs (tables)
# =========================
out_dir = "Dataset"
os.makedirs(out_dir, exist_ok=True)

summary_path = os.path.join(out_dir, "approach_evaluation_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Saved evaluation summary to '{summary_path}'")

results_path = os.path.join(out_dir, "Womens_clothing_review_results_enriched.csv")
df.to_csv(results_path, index=False)
print(f"Saved enriched review results to '{results_path}'")

# =========================
# Visualize Metrics (Charts)
# =========================
charts_dir = os.path.join(out_dir, "charts")
os.makedirs(charts_dir, exist_ok=True)

approaches = summary_df["Approach"].tolist()
metrics = ["Silhouette", "DaviesBouldin", "CalinskiHarabasz", "Dunn", "WCSS"]

# Helper to annotate bars
def annotate_bars(ax, values):
    for i, v in enumerate(values):
        if np.isnan(v):
            label = "NaN"
            y = 0
        else:
            label = f"{v:.3f}"
            y = v
        ax.text(i, y, label, ha="center", va="bottom", fontsize=9, rotation=0)

# 1) One bar chart per metric
for m in metrics:
    vals = summary_df[m].values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(approaches, vals)
    ax.set_title(f"{m} by Approach")
    ax.set_xlabel("Approach")
    ax.set_ylabel(m)
    annotate_bars(ax, vals)
    plt.tight_layout()
    fig_path = os.path.join(charts_dir, f"{m}_by_approach.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart: {fig_path}")

# 2) Combined dashboard: all metrics side-by-side (normalized for display except WCSS & DB inverted note)
# Normalize selected metrics to 0-1 for a comparative view (skip NaNs safely)
dash_metrics = ["Silhouette", "CalinskiHarabasz", "Dunn"]  # higher is better
norm_df = summary_df.copy()
for m in dash_metrics:
    col = norm_df[m].astype(float).values
    finite = np.isfinite(col)
    if finite.sum() > 0:
        mn, mx = np.nanmin(col[finite]), np.nanmax(col[finite])
        if mx > mn:
            col[finite] = (col[finite] - mn) / (mx - mn)
        else:
            col[finite] = 0.0
    norm_df[m] = col

fig, ax = plt.subplots(figsize=(9, 5))
bar_width = 0.25
x = np.arange(len(approaches))
for idx, m in enumerate(dash_metrics):
    ax.bar(x + idx*bar_width, norm_df[m].values, width=bar_width, label=m)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(approaches, rotation=0)
ax.set_ylabel("Normalized Score (0–1)")
ax.set_title("Normalized Comparison (Higher-is-better metrics)")
ax.legend()
plt.tight_layout()
dash_path = os.path.join(charts_dir, "normalized_comparison.png")
plt.savefig(dash_path, dpi=150)
plt.close(fig)
print(f"Saved chart: {dash_path}")
