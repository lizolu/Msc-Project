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
    silhouette_samples,
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
df = df[:100]  # keep small for quick testing; remove/adjust for full run
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

# Use generic names here; if you have a data-driven mapping to domain names, replace below.
cluster_name_map = {i: f"Cluster {i+1}" for i in range(kmeans.n_clusters)}
df["KMeans Cluster"] = df["KMeans Cluster"].map(cluster_name_map)

# =========================
# Metrics Functions (overall)
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
    Returns a dict of standard clustering metrics (overall).
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
# Encode Labels & Evaluate (overall)
# =========================
le_kw = LabelEncoder()
le_sem = LabelEncoder()
le_h = LabelEncoder()

labels_kmeans   = LabelEncoder().fit_transform(df["KMeans Cluster"])  # encode "Cluster 1" etc. to ints
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
# NEW: Per-cluster metrics per approach
# =========================
def per_cluster_metrics(X, labels):
    """
    Compute per-cluster metrics:
      - SilhouetteMean: mean silhouette score for samples in the cluster
      - DB_Component: Davies–Bouldin component for cluster i = max_j (S_i + S_j)/M_ij
      - WCSS_i: within-cluster sum of squares for cluster i
      - DunnLike: min centroid distance to another cluster / max intra-cluster diameter (across all)
      - Size: number of samples in cluster
    Returns DataFrame with one row per cluster label.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    uniq = np.unique(labels)

    # Silhouette per sample then mean per cluster (if more than 1 cluster)
    if len(uniq) >= 2:
        sil_samples = silhouette_samples(X, labels)
    else:
        sil_samples = np.full(len(labels), np.nan)

    # Precompute centroids, intra scatter (S_i), WCSS_i, diameter_i
    centroids = {}
    S = {}
    WCSS_i = {}
    diam = {}

    for c in uniq:
        mask = labels == c
        Xi = X[mask]
        if Xi.shape[0] == 0:
            centroids[c] = np.full(X.shape[1], np.nan)
            S[c] = np.nan
            WCSS_i[c] = np.nan
            diam[c] = np.nan
            continue
        centroid = Xi.mean(axis=0)
        centroids[c] = centroid
        # average L2 distance to centroid
        S[c] = float(np.mean(np.linalg.norm(Xi - centroid, axis=1))) if Xi.shape[0] > 0 else np.nan
        # WCSS_i
        WCSS_i[c] = float(((Xi - centroid) ** 2).sum())
        # diameter = max pairwise distance inside cluster
        if Xi.shape[0] >= 2:
            diam[c] = float(np.max(pdist(Xi)))
        else:
            diam[c] = 0.0

    # centroid distance matrix
    uniq_list = list(uniq)
    C = np.vstack([centroids[c] for c in uniq_list])
    if np.any(np.isnan(C)):
        M = np.full((len(uniq_list), len(uniq_list)), np.nan)
    else:
        M = squareform(pdist(C))

    # DB component per cluster
    DB_comp = {}
    for i_idx, ci in enumerate(uniq_list):
        # For a single cluster overall, set NaN
        if len(uniq_list) < 2 or np.isnan(M).all():
            DB_comp[ci] = np.nan
            continue
        # max_j (S_i + S_j)/M_ij, j != i (ignore 0 distance)
        ratios = []
        for j_idx, cj in enumerate(uniq_list):
            if ci == cj:
                continue
            denom = M[i_idx, j_idx]
            if denom > 0 and not np.isnan(denom):
                ratios.append((S[ci] + S[cj]) / denom)
        DB_comp[ci] = float(np.max(ratios)) if len(ratios) else np.nan

    # Dunn-like per cluster
    max_diameter = np.nanmax([diam[c] for c in uniq_list]) if len(uniq_list) > 0 else np.nan
    Dunn_like = {}
    for i_idx, ci in enumerate(uniq_list):
        if len(uniq_list) < 2 or np.isnan(M).all() or (max_diameter is None) or (max_diameter == 0) or np.isnan(max_diameter):
            Dunn_like[ci] = np.nan
            continue
        # nearest centroid distance to another cluster
        others = [M[i_idx, j_idx] for j_idx in range(len(uniq_list)) if j_idx != i_idx]
        nearest = np.nanmin(others) if len(others) else np.nan
        Dunn_like[ci] = float(nearest / max_diameter) if (nearest is not None and not np.isnan(nearest)) else np.nan

    # Build dataframe
    rows = []
    for c in uniq_list:
        mask = labels == c
        sil_mean = float(np.nanmean(sil_samples[mask])) if mask.any() else np.nan
        rows.append({
            "Cluster": c,
            "SilhouetteMean": sil_mean,
            "DB_Component": DB_comp[c] if c in DB_comp else np.nan,
            "WCSS_i": WCSS_i[c] if c in WCSS_i else np.nan,
            "DunnLike": Dunn_like[c] if c in Dunn_like else np.nan,
            "Size": int(mask.sum())
        })
    per_df = pd.DataFrame(rows)
    return per_df

# Compute per-cluster tables for each approach
per_kmeans   = per_cluster_metrics(review_embeddings, labels_kmeans)
per_keyword  = per_cluster_metrics(review_embeddings, labels_keyword)
per_semantic = per_cluster_metrics(review_embeddings, labels_semantic)
per_hybrid   = per_cluster_metrics(review_embeddings, labels_hybrid)

# Replace numeric cluster ids with readable names for non-KMeans approaches
# (Using inverse transform where available)
per_keyword["Cluster"]  = le_kw.inverse_transform(per_keyword["Cluster"].astype(int))
per_semantic["Cluster"] = le_sem.inverse_transform(per_semantic["Cluster"].astype(int))
per_hybrid["Cluster"]   = le_h.inverse_transform(per_hybrid["Cluster"].astype(int))
# KMeans clusters already named in df; map from encoded ints back to names for consistency
# Build mapping from encoded int -> name using the LabelEncoder fitted above for KMeans labels
le_kmeans = LabelEncoder().fit(df["KMeans Cluster"])
labels_kmeans_int = le_kmeans.transform(df["KMeans Cluster"])
per_kmeans["Cluster"] = le_kmeans.inverse_transform(per_kmeans["Cluster"].astype(int))

# =========================
# Save Outputs (tables)
# =========================
out_dir = "Dataset"
os.makedirs(out_dir, exist_ok=True)

summary_path = os.path.join(out_dir, "approach_evaluation_summary.csv")
summary_df.to_csv(summary_path, index=False)

per_kmeans.to_csv(os.path.join(out_dir, "per_cluster_metrics_kmeans.csv"), index=False)
per_keyword.to_csv(os.path.join(out_dir, "per_cluster_metrics_keyword.csv"), index=False)
per_semantic.to_csv(os.path.join(out_dir, "per_cluster_metrics_semantic.csv"), index=False)
per_hybrid.to_csv(os.path.join(out_dir, "per_cluster_metrics_hybrid.csv"), index=False)

print(f"Saved overall summary -> {summary_path}")
print("Saved per-cluster tables for each approach.")

# =========================
# Visualize Metrics (overall bar charts)
# =========================
charts_dir = os.path.join(out_dir, "charts")
os.makedirs(charts_dir, exist_ok=True)

approaches = summary_df["Approach"].tolist()
metrics_overall = ["Silhouette", "DaviesBouldin", "CalinskiHarabasz", "Dunn", "WCSS"]

def annotate_bars(ax, values):
    for i, v in enumerate(values):
        if np.isnan(v):
            label = "NaN"; y = 0
        else:
            label = f"{v:.3f}"; y = v
        ax.text(i, y, label, ha="center", va="bottom", fontsize=9)

# One bar chart per overall metric
for m in metrics_overall:
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

# =========================
# NEW: Line charts per approach with metrics as legend, clusters on x-axis
# =========================
def plot_per_cluster_lines(per_df, approach_name, out_dir):
    # Sort clusters by name for consistent x-axis
    per_df = per_df.copy()
    per_df = per_df.sort_values("Cluster")

    x_labels = per_df["Cluster"].astype(str).tolist()
    x = np.arange(len(x_labels))

    metric_cols = ["SilhouetteMean", "DB_Component", "WCSS_i", "DunnLike"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for m in metric_cols:
        ax.plot(x, per_df[m].values, marker='o', label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Per-Cluster Evaluation Metrics — {approach_name}")
    ax.legend(title="Metrics")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"per_cluster_metrics_{approach_name.lower()}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

paths = []
paths.append(plot_per_cluster_lines(per_kmeans,   "KMeans",   charts_dir))
paths.append(plot_per_cluster_lines(per_keyword,  "Keyword",  charts_dir))
paths.append(plot_per_cluster_lines(per_semantic, "Semantic", charts_dir))
paths.append(plot_per_cluster_lines(per_hybrid,   "Hybrid",   charts_dir))

for p in paths:
    print(f"Saved chart: {p}")
