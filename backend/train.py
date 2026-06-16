import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import joblib
import nltk

print("Downloading NLTK punkt data...")
nltk.download('punkt')
nltk.download('punkt_tab') # Might be needed in newer nltk versions

print("Loading dataset...")
df = pd.read_csv("dataset.csv")
df = df.drop_duplicates()

corpus = df["text"]
labels = df["label"]

print("Preprocessing text... (this may take a moment)")
processed_corpus = []
snowBall_stemmer = SnowballStemmer('english')

for sentence in corpus:
    tokens = word_tokenize(sentence)
    stemmed_tokens = []
    
    for word in tokens:
        stm = snowBall_stemmer.stem(word.lower())
        stemmed_tokens.append(stm)
        
    processed_sentence = " ".join(stemmed_tokens)
    processed_corpus.append(processed_sentence)

print("Vectorizing...")
vectorizer = TfidfVectorizer(ngram_range=(1,2))
x = vectorizer.fit_transform(processed_corpus)

print("Training model...")
model = MultinomialNB()
model.fit(x, labels)

print("Exporting models to .pkl files...")
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Success! model.pkl and vectorizer.pkl have been created.")
