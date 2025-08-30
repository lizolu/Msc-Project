import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
import numpy as np

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load dataset
df = pd.read_csv("Dataset/Womens Clothing E-Commerce Reviews.csv")
df = df[~df['Review Text'].isnull()]

# Define category keywords
cluster_keywords = {
    "Style and Aesthetics": ["stylish", "trendy", "fashionable", "chic", "modern", "elegant", "cute", "beautiful", "design", "pattern", "color", "print", "classy", "look", "style", "flattering", "aesthetic", "unique", "versatile", "sleek"],
    "Fit and Sizing": ["tight", "loose", "fitted", "oversized", "true to size", "small", "big", "fit", "sizing", "proportion", "snug", "comfortable fit", "perfect fit", "length", "short", "long", "size", "waist", "hips", "shoulder"],
    "Fabric and Material Quality": ["soft", "rough", "scratchy", "smooth", "see-through", "durable", "delicate", "stretchy", "fabric", "material", "quality", "texture", "thick", "thin", "luxurious", "lightweight", "heavy", "breathable", "synthetic", "cotton"],
    "Comfort and Wearability": ["comfortable", "uncomfortable", "lightweight", "breathable", "warm", "cool", "easy to wear", "soft", "itchy", "stretchy", "relaxed", "casual", "practical", "movable", "airy", "cozy", "snug", "restrictive", "functional", "pleasant"],
    "Occasion and Use Case": ["work", "office", "wedding", "party", "casual", "formal", "evening", "daywear", "holiday", "vacation", "beach", "gym", "event", "ceremony", "travel", "special occasion", "weekend", "everyday", "festive", "outing"],
    "Price-Value Perception": ["affordable", "expensive", "worth", "overpriced", "cheap", "value", "price", "reasonable", "budget", "deal", "cost", "investment", "bargain", "quality for price", "overvalued", "economical", "pricy", "low-cost", "steal", "costly"]
}

# Preprocessing
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return words

df["Processed Review"] = df["Review Text"].apply(lambda x: clean_text(str(x)))

# Load BERT-based model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for category keyword lists
category_sentences = {cat: " ".join(words) for cat, words in cluster_keywords.items()}
category_embeddings = {cat: model.encode(desc, convert_to_tensor=True) for cat, desc in category_sentences.items()}

# Function to assign category based on hybrid scoring
def assign_category(review_words, original_text):
    scores = {}
    review_embedding = model.encode(original_text, convert_to_tensor=True)
    for cat, keywords in cluster_keywords.items():
        stemmed_keywords = [stemmer.stem(w.lower()) for w in keywords]
        keyword_score = sum(word in stemmed_keywords for word in review_words) / len(stemmed_keywords)
        semantic_score = util.pytorch_cos_sim(review_embedding, category_embeddings[cat]).item()
        combined_score = (keyword_score * 0.5) + (semantic_score * 0.5)
        scores[cat] = combined_score
    return max(scores, key=scores.get)

df["Hybrid Category"] = df.apply(lambda row: assign_category(row["Processed Review"], row["Review Text"]), axis=1)

# === Performance Evaluation Section ===
# Generate embeddings for reviews
review_embeddings = model.encode(df["Review Text"].tolist(), convert_to_tensor=False)

# Run KMeans with 6 clusters
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df["KMeans Cluster"] = kmeans.fit_predict(review_embeddings)

# Metrics
silhouette = silhouette_score(review_embeddings, df["KMeans Cluster"])
davies_bouldin = davies_bouldin_score(review_embeddings, df["KMeans Cluster"])
calinski_harabasz = calinski_harabasz_score(review_embeddings, df["KMeans Cluster"])
wcss = kmeans.inertia_

# Dunn Index implementation
def dunn_index(X, labels):
    distances = squareform(pdist(X))
    unique_clusters = np.unique(labels)
    intra = []
    for c in unique_clusters:
        points = np.where(labels == c)[0]
        if len(points) > 1:
            intra.append(np.max(distances[np.ix_(points, points)]))
    max_intra = np.max(intra)
    inter = []
    for i in range(len(unique_clusters)):
        for j in range(i+1, len(unique_clusters)):
            points_i = np.where(labels == unique_clusters[i])[0]
            points_j = np.where(labels == unique_clusters[j])[0]
            inter.append(np.min(distances[np.ix_(points_i, points_j)]))
    min_inter = np.min(inter)
    return min_inter / max_intra

dunn = dunn_index(review_embeddings, df["KMeans Cluster"])

# Print evaluation results
print("\n=== Clustering Performance Metrics ===")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
print(f"Dunn Index: {dunn:.4f}")
print(f"Within-Cluster Sum of Squares (WCSS): {wcss:.4f}")

# === Sample results ===
pd.set_option('display.max_colwidth', None)
for label in df["Hybrid Category"].unique():
    print(f"\n=== {label} ===")
    print(df[df["Hybrid Category"] == label]["Review Text"].head(3).to_string(index=False))

# Encode Hybrid Category into numeric labels
label_encoder = LabelEncoder()
df["Hybrid_Label"] = label_encoder.fit_transform(df["Hybrid Category"])

# Generate embeddings for reviews
review_embeddings = model.encode(df["Review Text"].tolist(), convert_to_tensor=False)

# === Performance Metrics for Hybrid Category ===
# Silhouette Score
silhouette = silhouette_score(review_embeddings, df["Hybrid_Label"])

# Davies–Bouldin Index
davies_bouldin = davies_bouldin_score(review_embeddings, df["Hybrid_Label"])

# Calinski–Harabasz Index
calinski_harabasz = calinski_harabasz_score(review_embeddings, df["Hybrid_Label"])

# Dunn Index
def dunn_index(X, labels):
    distances = squareform(pdist(X))
    unique_clusters = np.unique(labels)
    intra = []
    for c in unique_clusters:
        points = np.where(labels == c)[0]
        if len(points) > 1:
            intra.append(np.max(distances[np.ix_(points, points)]))
    max_intra = np.max(intra)
    inter = []
    for i in range(len(unique_clusters)):
        for j in range(i+1, len(unique_clusters)):
            points_i = np.where(labels == unique_clusters[i])[0]
            points_j = np.where(labels == unique_clusters[j])[0]
            inter.append(np.min(distances[np.ix_(points_i, points_j)]))
    min_inter = np.min(inter)
    return min_inter / max_intra

dunn = dunn_index(review_embeddings, df["Hybrid_Label"])

# WCSS (Within-Cluster Sum of Squares)
# Compute centroids based on Hybrid labels
def compute_wcss(X, labels):
    wcss = 0
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        wcss += ((cluster_points - centroid) ** 2).sum()
    return wcss

review_embeddings_np = np.array(review_embeddings)
wcss = compute_wcss(review_embeddings_np, df["Hybrid_Label"].values)

# Print results
print("\n=== Hybrid Category Performance Metrics ===")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
print(f"Dunn Index: {dunn:.4f}")
print(f"Within-Cluster Sum of Squares (WCSS): {wcss:.4f}")

df.to_csv(".\Dataset\Women's clothing review results.csv")