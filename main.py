from textblob import TextBlob
from langdetect import detect

# 🔹 Sample blog-style paragraph (inbuilt)
sample_text = """
Artificial Intelligence (AI) and Data Science are transforming industries rapidly.
This blog explores how machine learning models are improving healthcare, finance, and education.
The future looks promising and full of exciting innovations.
"""

# 🔸 Create TextBlob object
blob = TextBlob(sample_text)

# 📊 Sentiment Analysis
print("=== Sentiment Analysis ===")
print("Polarity     :", blob.sentiment.polarity)
print("Subjectivity :", blob.sentiment.subjectivity)

# 🔍 Noun Phrase Extraction
print("\n=== Noun Phrases ===")
for np in blob.noun_phrases:
    print("-", np)

# 🌐 Language Detection
try:
    language = detect(sample_text)
    print("\n=== Language Detection ===")
    print("Detected Language:", language)

    # 🔁 Translation if needed
    if language != 'en':
        try:
            translated = blob.translate(to='en')
            print("\n=== Translated Text ===")
            print(translated)
        except Exception as e:
            print("Translation failed:", e)

except Exception as e:
    print("Language detection failed:", e)
