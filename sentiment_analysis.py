import pandas as pd
import numpy as np
import tensorflow as tf
import time

from sklearn import svm
from nn_model import nn_model
from eval import evaluate_model
from preprocessing import *
from vectorizers import *
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re

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

random_state = 42

# Amazon dataset
amazon = pd.read_csv(r".\data\amazon_electronics_review.csv", sep='\t', index_col=[0])
amazon = amazon[~amazon['reviewText'].isna()]
amazon['Label'] = amazon['overall'].apply(lambda x: 'negative' if x<3 else 'positive' if x>3 else 'neutral')

amazon = pd.concat([amazon[amazon['Label']=='negative'][:17000],
                    amazon[amazon['Label']=='positive'][:17000]], ignore_index=True)

amazon.rename(columns={'reviewText': 'Reviews'}, inplace=True)
#amazon = amazon.sample(frac=1, random_state=random_state)

# IMDB dataset
imdb = pd.read_csv(r".\data\IMDB.csv", sep='\t')
#imdb = imdb.sample(frac=1, random_state=random_state)

# Cherwell dataset
cherwell = pd.read_csv('.\data\Cherwell.csv', sep='\t')
cherwell.rename(columns={'sentiment': 'Label', 'Inbound Details (Masked)': 'Reviews'}, inplace=True)
cherwell = cherwell[cherwell['Label'].isin(['Positive','Negative'])].reset_index(drop=True)
cherwell['Label'] = cherwell['Label'].apply(lambda x: x.lower())
#cherwell = cherwell[:200]


datasets = {
            'amazon': amazon,
            'imdb': imdb,
            'cherwell': cherwell
            }

# for each data set run model
for dataset_name, dataset in datasets.items():
    print(f"="*50)
    print(f"Dataset: {dataset_name}")
    print(f"="*50)
    raw_dataset = dataset.copy()
    raw_train, raw_test = train_test_split(raw_dataset, test_size=0.2, random_state=random_state)

    # Prep
    dataset['Reviews'] = dataset['Reviews'].apply(lambda x: remove_stop_words(x))
    dataset['Reviews'] = dataset['Reviews'].apply(lambda x: deEmojify(x))
    dataset['Reviews'] = dataset['Reviews'].apply(lambda x: stem_words(x))

    #dataset = dataset.sample(frac=1, random_state=random_state)
    train, test = train_test_split(dataset, test_size=0.2, random_state=random_state)
    nn_train_label = train['Label'].map({'positive': 1, 'negative':0})
    nn_train_label = tf.cast(nn_train_label, tf.float32)


    # TFIDF vectorizer
    train_vectors, test_vectors = get_tfidf_vectors(train['Reviews'], test['Reviews'])
    tfidf_train_vectors, tfidf_test_vectors = train_vectors, test_vectors

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear', random_state=random_state)
    classifier_linear.fit(train_vectors, train['Label'])
    prediction_linear = classifier_linear.predict(test_vectors)

    # Evaluate SVM model performance
    evaluate_model(test['Label'], prediction_linear)

    # Train neural network model
    model = nn_model(train_vectors, nn_train_label)
    tf.keras.utils.plot_model(model, "model_architecture.png", show_shapes=True)
    pred_nn = model.predict(test_vectors)
    pred_nn = np.rint(pred_nn).tolist()
    pred_nn = [
        x
        for xs in pred_nn
        for x in xs
    ]
    pred_nn = pd.Series(pred_nn).map({1.0: 'positive', 0.0:'negative'})
    print("Neural network ...")

    # Evaluate NN model performance
    evaluate_model(test['Label'], pred_nn)


    # Count vectorizer
    train_vectors, test_vectors = get_count_vectors(train['Reviews'], test['Reviews'])

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear', random_state=random_state)
    classifier_linear.fit(train_vectors, train['Label'])
    prediction_linear = classifier_linear.predict(test_vectors)

    # Evaluate model performance
    evaluate_model(test['Label'], prediction_linear)

    # Train neural network model
    model = nn_model(train_vectors, nn_train_label)
    pred_nn = model.predict(test_vectors)
    pred_nn = np.rint(pred_nn).tolist()
    pred_nn = [
        x
        for xs in pred_nn
        for x in xs
    ]
    pred_nn = pd.Series(pred_nn).map({1.0: 'positive', 0.0:'negative'})
    print("Neural network ...")

    # Evaluate NN model performance
    evaluate_model(test['Label'], pred_nn)


    # Hashing vectorizer
    train_vectors, test_vectors = get_hashing_vectors(train['Reviews'], test['Reviews'])
    
    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear', random_state=random_state)
    classifier_linear.fit(train_vectors, train['Label'])
    prediction_linear = classifier_linear.predict(test_vectors)

    # Evaluate model performance
    evaluate_model(test['Label'], prediction_linear)

    # Train neural network model
    model = nn_model(train_vectors, nn_train_label)
    pred_nn = model.predict(test_vectors)
    pred_nn = np.rint(pred_nn).tolist()
    pred_nn = [
        x
        for xs in pred_nn
        for x in xs
    ]
    pred_nn = pd.Series(pred_nn).map({1.0: 'positive', 0.0:'negative'})
    print("Neural network ...")

    # Evaluate NN model performance
    evaluate_model(test['Label'], pred_nn)


    # Glove vectorizer
    train_vectors = get_glove_vectors(train['Reviews'])
    test_vectors = get_glove_vectors(test['Reviews'])

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear', random_state=random_state)
    classifier_linear.fit(train_vectors, train['Label'])
    prediction_linear = classifier_linear.predict(test_vectors)

    # Evaluate SVM model performance
    evaluate_model(test['Label'], prediction_linear)

    # Train neural network model
    model = nn_model(train_vectors, nn_train_label)
    pred_nn = model.predict(test_vectors)
    pred_nn = np.rint(pred_nn).tolist()
    pred_nn = [
        x
        for xs in pred_nn
        for x in xs
    ]
    pred_nn = pd.Series(pred_nn).map({1.0: 'positive', 0.0:'negative'})
    print("Neural network ...")

    # Evaluate NN model performance
    evaluate_model(test['Label'], pred_nn)


    # word2vec vectorizer
    train_vectors, test_vectors = get_word2vec_vectors(train['Reviews'], test['Reviews'])

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear', random_state=random_state)
    classifier_linear.fit(train_vectors, train['Label'])
    prediction_linear = classifier_linear.predict(test_vectors)

    # Evaluate SVM model performance
    evaluate_model(test['Label'], prediction_linear)

    # Train neural network model
    model = nn_model(train_vectors, nn_train_label)
    pred_nn = model.predict(test_vectors)
    pred_nn = np.rint(pred_nn).tolist()
    pred_nn = [
        x
        for xs in pred_nn
        for x in xs
    ]
    pred_nn = pd.Series(pred_nn).map({1.0: 'positive', 0.0:'negative'})
    print("Neural network ...")

    # Evaluate NN model performance
    evaluate_model(test['Label'], pred_nn)
    

    # Get BERT embeddings for the text data
    bert_train_vectors = get_bert_embeddings(raw_train['Reviews'].to_list())
    bert_test_vectors = get_bert_embeddings(raw_test['Reviews'].to_list())

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.LinearSVC(random_state=random_state)
    classifier_linear.fit(bert_train_vectors, train['Label'])
    prediction_linear = classifier_linear.predict(bert_test_vectors)

    # Evaluate model performance
    evaluate_model(test['Label'], prediction_linear)

    # Train neural network model
    model = nn_model(bert_train_vectors, nn_train_label)
    pred_nn = model.predict(bert_test_vectors)
    pred_nn = np.rint(pred_nn).tolist()
    pred_nn = [
        x
        for xs in pred_nn
        for x in xs
    ]
    pred_nn = pd.Series(pred_nn).map({1.0: 'positive', 0.0:'negative'})
    print("Neural network ...")

    # Evaluate NN model performance
    evaluate_model(test['Label'], pred_nn)


    # Get hybrid embeddings
    # Concatenate TF-IDF vectors with BERT embeddings
    hybrid_train_embedding = np.hstack((tfidf_train_vectors.toarray(), bert_train_vectors))
    hybrid_test_embedding = np.hstack((tfidf_test_vectors.toarray(), bert_test_vectors))

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.LinearSVC(random_state=random_state)
    classifier_linear.fit(hybrid_train_embedding, train['Label'])
    prediction_linear = classifier_linear.predict(hybrid_test_embedding)

    # Evaluate model performance
    evaluate_model(test['Label'], prediction_linear)

    # Train neural network model
    model = nn_model(hybrid_train_embedding, nn_train_label)
    pred_nn = model.predict(hybrid_test_embedding)
    pred_nn = np.rint(pred_nn).tolist()
    pred_nn = [
        x
        for xs in pred_nn
        for x in xs
    ]
    pred_nn = pd.Series(pred_nn).map({1.0: 'positive', 0.0:'negative'})
    print("Neural network ...")

    # Evaluate NN model performance
    evaluate_model(test['Label'], pred_nn)