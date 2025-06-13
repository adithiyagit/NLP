import nltk

# Ensure necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Load text
with open('sample_text.txt', 'r') as f:
    text = f.read()

# NLTK Tokenization
tokens = TreebankWordTokenizer().tokenize(text)
print("\n[NLTK] Tokens:")
print(tokens)

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("\n[NLTK] After Stopword Removal:")
print(filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("\n[NLTK] After Stemming:")
print(stemmed_tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\n[NLTK] After Lemmatization:")
print(lemmatized_tokens)

# ---- spaCy Section ----

import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Reuse the same text
doc = nlp(text)

# spaCy Tokenization
tokens = [token.text for token in doc]
print("\n[spaCy] Tokens:")
print(tokens)

# Stopword Removal
filtered_tokens_spacy = [token.text for token in doc if not token.is_stop]
print("\n[spaCy] After Stopword Removal:")
print(filtered_tokens_spacy)

# Lemmatization
lemmatized_tokens_spacy = [token.lemma_ for token in doc if not token.is_stop]
print("\n[spaCy] After Lemmatization:")
print(lemmatized_tokens_spacy)

# Note: spaCy does NOT use stemming by design.
