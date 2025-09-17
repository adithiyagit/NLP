import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Sample dataset
texts = [
    'I love this movie',
    'This movie is terrible',
    'I really enjoyed this film',
    'This film is awful',
    'What a fantastic experience',
    'I hated this film',
    'This was a great movie',
    'The film was not good',
    'I am very happy with this movie',
    'I am disappointed with this film'
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative sentiment

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# ==========================
# Bag of Words Model
# ==========================
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

model_bow = MultinomialNB()
model_bow.fit(X_train_bow, y_train)
pred_bow = model_bow.predict(X_test_bow)

# Evaluation - BoW
print("----- Bag of Words -----")
print("Accuracy:", accuracy_score(y_test, pred_bow))
print("Classification Report:\n", classification_report(y_test, pred_bow))

# Confusion Matrix - BoW
cm_bow = confusion_matrix(y_test, pred_bow)
sns.heatmap(cm_bow, annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix - BoW")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ==========================
# TF-IDF Model
# ==========================
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model_tfidf = MultinomialNB()
model_tfidf.fit(X_train_tfidf, y_train)
pred_tfidf = model_tfidf.predict(X_test_tfidf)

# Evaluation - TF-IDF
print("----- TF-IDF -----")
print("Accuracy:", accuracy_score(y_test, pred_tfidf))
print("Classification Report:\n", classification_report(y_test, pred_tfidf))

# Confusion Matrix - TF-IDF
cm_tfidf = confusion_matrix(y_test, pred_tfidf)
sns.heatmap(cm_tfidf, annot=True, cmap='Greens', fmt='d')
plt.title("Confusion Matrix - TF-IDF")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()