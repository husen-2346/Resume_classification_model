# ==============================
# IMPORT REQUIRED LIBRARIES
# ==============================

# Tokenization for splitting sentences into words
from nltk.tokenize import word_tokenize

# Snowball stemmer for reducing words to root form
from nltk.stem import SnowballStemmer

# TF-IDF vectorizer for converting text into numerical vectors
from sklearn.feature_extraction.text import TfidfVectorizer

# Naive Bayes algorithm for text classification
from sklearn.naive_bayes import MultinomialNB

# Pandas for dataset handling
import pandas as pd

# Google Drive integration for Colab
from google.colab import drive


# ==============================
# MOUNT GOOGLE DRIVE
# ==============================

drive.mount('/content/drive')


# ==============================
# LOAD DATASET
# ==============================

# Read CSV dataset
df = pd.read_csv("/content/drive/MyDrive/dataset.csv")

# Remove duplicate rows from dataset
df = df.drop_duplicates()


# ==============================
# EXTRACT FEATURES AND LABELS
# ==============================

# Resume/job text data
corpus = df["text"]

# Corresponding categories/labels
labels = df["label"]


# ==============================
# NLP PREPROCESSING
# ==============================

# Store processed text here
processed_corpus = []

# Initialize Snowball Stemmer
snowball_stemmer = SnowballStemmer('english')


# Process each sentence in corpus
for sentence in corpus:

    # Convert sentence into tokens(words)
    tokens = word_tokenize(sentence)

    # Store processed words
    stemmed_tokens = []

    # Process each token
    for word in tokens:

        # Convert word to lowercase
        word = word.lower()

        # Apply stemming
        stemmed_word = snowball_stemmer.stem(word)

        # Add processed word to list
        stemmed_tokens.append(stemmed_word)

    # Join processed words back into sentence
    processed_sentence = " ".join(stemmed_tokens)

    # Add processed sentence to final corpus
    processed_corpus.append(processed_sentence)


# ==============================
# TEXT VECTORIZATION USING TF-IDF
# ==============================

# Convert text into TF-IDF numerical vectors
# ngram_range=(1,2) means:
# - Unigrams(single words)
# - Bigrams(two-word combinations)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Fit and transform processed text
X = vectorizer.fit_transform(processed_corpus)


# ==============================
# MODEL TRAINING - NAIVE BAYES
# ==============================

# Initialize Multinomial Naive Bayes model
model = MultinomialNB()

# Train model using TF-IDF vectors and labels
model.fit(X, labels)


# ==============================
# MODEL IS NOW TRAINED
# ==============================

print("Model training completed successfully!")

# ==============================
# NEW INPUT DATA FOR PREDICTION
# ==============================

# Example resume/skill text
new_data = "react nodejs html css"


# ==============================
# PREPROCESS NEW INPUT TEXT
# ==============================

# Store processed input text
processed_new_data = []

# Convert sentence into tokens(words)
tokens = word_tokenize(new_data)

# Store processed tokens
stemmed_tokens = []

# Process each token
for word in tokens:

    # Convert word to lowercase
    word = word.lower()

    # Apply stemming
    stemmed_word = snowball_stemmer.stem(word)

    # Store processed word
    stemmed_tokens.append(stemmed_word)

# Join processed tokens into sentence
processed_sentence = " ".join(stemmed_tokens)

# Add processed sentence to list
processed_new_data.append(processed_sentence)


# ==============================
# CONVERT TEXT INTO TF-IDF VECTOR
# ==============================

# IMPORTANT:
# Use transform() only
# Do NOT use fit_transform() here
#
# Reason:
# We must use the SAME vocabulary learned during training

new_X = vectorizer.transform(processed_new_data)


# ==============================
# PREDICT CATEGORY USING MODEL
# ==============================

prediction = model.predict(new_X)


# ==============================
# DISPLAY RESULT
# ==============================

print("Input Text:")
print(new_data)

print("\nPredicted Category:")
print(prediction[0])