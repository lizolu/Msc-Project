import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load data
df = pd.read_csv("Dataset/Womens Clothing E-Commerce Reviews.csv")
df = df[~df['Review Text'].isnull()]

# Cluster keyword dictionary
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
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # remove punctuation/numbers
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return words

df["Processed Review"] = df["Review Text"].apply(lambda x: clean_text(str(x)))

# Assign cluster based on keyword matching
def assign_category(words):
    scores = {cat: 0 for cat in cluster_keywords.keys()}
    for cat, kw_list in cluster_keywords.items():
        stemmed_keywords = [stemmer.stem(w.lower()) for w in kw_list]
        scores[cat] = sum(word in stemmed_keywords for word in words)
    # Assign to category with highest score
    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else "Uncategorized"

df["Category"] = df["Processed Review"].apply(assign_category)

# Show sample results
pd.set_option('display.max_colwidth', None)
for label in df["Category"].unique():
    print(f"\n=== {label} ===")
    print(df[df["Category"] == label]["Review Text"].head(3).to_string(index=False))
