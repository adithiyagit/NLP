import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 1: Load dataset
df = pd.read_csv("sentimentdataset.csv")  # Replace with your file

# Step 2: Clean and preprocess
df = df[['Text', 'Sentiment']].dropna()
df['Text'] = df['Text'].str.strip()
df['Sentiment'] = df['Sentiment'].str.strip()

# Step 3: Remove underrepresented classes
label_counts = df['Sentiment'].value_counts()
valid_classes = label_counts[label_counts >= 2].index.tolist()
df = df[df['Sentiment'].isin(valid_classes)]

# Step 4: Encode labels
label_encoder = LabelEncoder()
df['SentimentEncoded'] = label_encoder.fit_transform(df['Sentiment'])

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'], df['SentimentEncoded'], test_size=0.2, random_state=42, stratify=df['SentimentEncoded']
)

# Step 6: Pipelines
cbow_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

tfidf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Step 7: Train models
cbow_pipeline.fit(X_train, y_train)
tfidf_pipeline.fit(X_train, y_train)

# Step 8: Predict
cbow_preds = cbow_pipeline.predict(X_test)
tfidf_preds = tfidf_pipeline.predict(X_test)

# Step 9: Evaluate
used_labels = sorted(list(set(y_test)))
used_class_names = label_encoder.inverse_transform(used_labels)

print("📊 Bag-of-Words Accuracy:", accuracy_score(y_test, cbow_preds))
print("📊 TF-IDF Accuracy:", accuracy_score(y_test, tfidf_preds))

print("\n📝 Classification Report (Bag-of-Words):")
print(classification_report(y_test, cbow_preds, labels=used_labels, target_names=used_class_names))

print("\n📝 Classification Report (TF-IDF):")
print(classification_report(y_test, tfidf_preds, labels=used_labels, target_names=used_class_names))

# Step 10: WordCloud
text_data = " ".join(df['Text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud of All Text Data", fontsize=14)
plt.tight_layout()
plt.show()

# Step 11: Confusion Matrix Heatmaps
def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=used_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=used_class_names,
                yticklabels=used_class_names, cmap="YlGnBu")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_confusion(y_test, cbow_preds, "Bag-of-Words")
plot_confusion(y_test, tfidf_preds, "TF-IDF")
