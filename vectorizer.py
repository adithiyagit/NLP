import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Sample text data
documents = [
    "Data science is a growing field with many opportunities.",
    "Machine learning is a part of data science.",
    "Natural language processing is important for data analysis."
]

# Bag-of-Wordsgit 
cv = CountVectorizer()
bow = cv.fit_transform(documents)
bow_words = dict(zip(cv.get_feature_names_out(), bow.sum(axis=0).A1))

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)
tfidf_words = dict(zip(tfidf.get_feature_names_out(), tfidf_matrix.sum(axis=0).A1))

# Cosine Similarity
bow_sim = cosine_similarity(bow.toarray())
tfidf_sim = cosine_similarity(tfidf_matrix.toarray())

print("BoW Cosine Similarity Matrix:\n", bow_sim)
print("\nTF-IDF Cosine Similarity Matrix:\n", tfidf_sim)

# --- Heatmaps ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(bow_sim, annot=True, cmap="Blues", xticklabels=["Doc1", "Doc2", "Doc3"], yticklabels=["Doc1", "Doc2", "Doc3"])
plt.title("BoW Cosine Similarity")

plt.subplot(1, 2, 2)
sns.heatmap(tfidf_sim, annot=True, cmap="Greens", xticklabels=["Doc1", "Doc2", "Doc3"], yticklabels=["Doc1", "Doc2", "Doc3"])
plt.title("TF-IDF Cosine Similarity")
plt.tight_layout()
plt.show()

# --- Word Clouds ---
WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bow_words).to_image().show()
WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_words).to_image().show()
