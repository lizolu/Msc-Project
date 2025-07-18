import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re



# 2. Load your CSV file
df = pd.read_csv("Dataset/Womens Clothing E-Commerce Reviews.csv")
# remove nulls
df = df[~df['Review Text'].isnull()]

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def remove_stop_words(x):
    text = ' '.join([word for word in x.split() if word not in stop_words])
    return text

def stem_words(x):
    text = ' '.join([stemmer.stem(word.lower()) for word in x.split()])
    return text

def preprocess(text):
    text = deEmojify(text)
    text= remove_stop_words(text)
    text = stem_words(text)
    return text

# 5. Apply preprocessing to "Review Text"
df["Processed Review"] = df["Review Text"].apply(lambda x: preprocess(x))

# 6. Save or view result
# df[["Review Text", "Processed Review"]].to_csv("processed_reviews.csv", index=False)
df[["Review Text", "Processed Review"]].head()

print(df)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_df=0.8, min_df=5)
X = vectorizer.fit_transform(df['Processed Review'])

# 2. K-Means Clustering (e.g., 5 clusters - you can change this)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 3. Show sample results
for i in range(k):
    print(f"\nCluster {i} Sample Reviews:")
    print(df[df['Cluster'] == i]['Review Text'].head(3).to_string(index=False))
import numpy as np

terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

print("\nTop terms per cluster:")
for i in range(k):
    print(f"Cluster {i}:")
    print(", ".join([terms[ind] for ind in order_centroids[i, :10]]))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# 1. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1,2))  # includes bigrams
X = vectorizer.fit_transform(df['Processed Review'])

# 2. K-Means Clustering (start with 5 clusters for interpretability)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 3. Extract Top Keywords per Cluster
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

cluster_keywords = []
print("\n=== Top keywords per cluster ===")
for i in range(k):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    cluster_keywords.append(top_terms)
    print(f"Cluster {i}: {', '.join(top_terms)}")
cluster_labels = {
    0: "Fit and Size Complaints",
    1: "Positive Feedback / Style Praise",
    2: "Material Quality and Texture",
    3: "Trendy or Elegant Styles",
    4: "Service or Shipping Concerns"
}

df["Cluster Label"] = df["Cluster"].map(cluster_labels)
