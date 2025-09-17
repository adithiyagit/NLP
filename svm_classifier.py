# Step 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Step 2: Define the dataset
corpus = [
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

# Labels: 1 = Positive, 0 = Negative
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Step 3: Split the dataset
X_train_text, X_test_text, y_train, y_test = train_test_split(
    corpus, labels, test_size=0.2, random_state=42
)

# Step 4: Define a function to train and evaluate a model
def train_and_evaluate(vectorizer, title):
    # Vectorize text
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Train SVM
    model = SVC(kernel='linear', C=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    print(f"\n--- {title} ---")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.grid(False)
    plt.show()

    return model

# Step 5: Train and visualize for Bag of Words
bow_vectorizer = CountVectorizer()
bow_model = train_and_evaluate(bow_vectorizer, "Bag of Words (BoW)")

# Step 6: Train and visualize for TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_model = train_and_evaluate(tfidf_vectorizer, "TF-IDF")
