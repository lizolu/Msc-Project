import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, util

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
    "Price–Value Perception": ["affordable", "expensive", "worth", "overpriced", "cheap", "value", "price", "reasonable", "budget", "deal", "cost", "investment", "bargain", "quality for price", "overvalued", "economical", "pricy", "low-cost", "steal", "costly"]
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
    review_text_clean = " ".join(review_words)
    
    # Encode review for semantic similarity
    review_embedding = model.encode(original_text, convert_to_tensor=True)
    
    for cat, keywords in cluster_keywords.items():
        # Keyword score (exact/stemmed match count)
        stemmed_keywords = [stemmer.stem(w.lower()) for w in keywords]
        keyword_score = sum(word in stemmed_keywords for word in review_words) / len(stemmed_keywords)
        
        # Semantic similarity score
        semantic_score = util.pytorch_cos_sim(review_embedding, category_embeddings[cat]).item()
        
        # Weighted combination (tune weights for balance)
        combined_score = (keyword_score * 0.5) + (semantic_score * 0.5)
        scores[cat] = combined_score
    
    # Pick category with highest combined score
    return max(scores, key=scores.get)

df["Hybrid Category"] = df.apply(lambda row: assign_category(row["Processed Review"], row["Review Text"]), axis=1)

# Show sample results
pd.set_option('display.max_colwidth', None)
for label in df["Hybrid Category"].unique():
    print(f"\n=== {label} ===")
    print(df[df["Hybrid Category"] == label]["Review Text"].head(3).to_string(index=False))
