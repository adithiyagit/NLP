import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy


# Load resources
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Read text
with open('sample_text.txt', 'r') as file:
    text = file.read()

print("Original Text:\n", text)

# --------------------------------------------
# 1. WORD TOKENIZATION (NLTK)
# --------------------------------------------
tokens = TreebankWordTokenizer().tokenize(text)
print("\n1. Tokenized Words:")
print(tokens)

# --------------------------------------------
# 2. STOPWORD REMOVAL (NLTK)
# --------------------------------------------
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("\n2. After Stopword Removal:")
print(filtered_tokens)

# --------------------------------------------
# 3. STEMMING (NLTK)
# --------------------------------------------
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("\n3. After Stemming:")
print(stemmed_words)

# --------------------------------------------
# 4. LEMMATIZATION (spaCy)
# --------------------------------------------
doc = nlp(text)
lemmatized_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
print("\n4. After Lemmatization (spaCy):")
print(lemmatized_words)
