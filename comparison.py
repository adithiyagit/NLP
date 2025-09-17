import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import re
import string
import os

print("Current directory:", os.getcwd())

# Load the dataset
# Use 'text' as input and 'rating' as label
CSV_PATH = r"C:\Users\T470S\Documents\DeskTop\NLP\combined\JAL_tripadvisor_reviews.csv"
df = pd.read_csv(CSV_PATH)

print("Columns:", df.columns)

TEXT_COLUMN = 'text'      # Review text
LABEL_COLUMN = 'rating'   # Review rating (1-5)

df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

# Encode the rating labels as integers (if not already)
le = LabelEncoder()
df[LABEL_COLUMN] = le.fit_transform(df[LABEL_COLUMN])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN], df[LABEL_COLUMN], test_size=0.2, random_state=42, stratify=df[LABEL_COLUMN]
)

vectorizers = {
    "Bag of Words": CountVectorizer(),
    "TF-IDF": TfidfVectorizer()
}

models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42)
}

for vec_name, vectorizer in vectorizers.items():
    print(f"\n=== Using Vectorizer: {vec_name} ===")
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_.astype(str))) 