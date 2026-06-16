from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import SnowballStemmer
except ImportError:
    pass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Serve frontend static files from sibling frontend/ directory
app = Flask(__name__, static_folder='../frontend', static_url_path='')
# Enable CORS for frontend integration
CORS(app)

@app.route('/')
def home():
    return app.send_static_file('index.html')

try:
    # Initialize stemmer just like in the notebook
    snowBall_stemmer = SnowballStemmer("english")
except Exception:
    snowBall_stemmer = None

# Auto-train if models don't exist
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    print("Models not found! Automatically training from dataset.csv... Please wait a few seconds.")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('punkt_tab')
        
        df = pd.read_csv("dataset.csv")
        df = df.drop_duplicates()
        
        corpus = df["text"]
        labels = df["label"]
        
        processed_corpus = []
        for sentence in corpus:
            tokens = word_tokenize(sentence)
            stemmed_tokens = []
            for word in tokens:
                if snowBall_stemmer:
                    stm = snowBall_stemmer.stem(word.lower())
                else:
                    stm = word.lower()
                stemmed_tokens.append(stm)
            processed_corpus.append(" ".join(stemmed_tokens))
            
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        x = vectorizer.fit_transform(processed_corpus)
        
        model = MultinomialNB()
        model.fit(x, labels)
        
        joblib.dump(model, "model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
        print("Training complete! Models saved.")
    except Exception as e:
        print(f"Error during auto-training: {e}")

# 1. Load models at startup
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    model_loaded = True
except Exception as e:
    model = None
    vectorizer = None
    model_loaded = False
    print(f"Warning: Model not found ({e}). Make sure to export model.pkl and vectorizer.pkl.")

try:
    # Initialize stemmer just like in the notebook
    snowBall_stemmer = SnowballStemmer("english")
except Exception:
    snowBall_stemmer = None

# 2. Reusable preprocessing function matching the notebook logic
def preprocess_text(text):
    if not snowBall_stemmer:
        return text
    
    # Tokenization
    tokens = word_tokenize(text)
    stemmed_tokens = []
    
    # Snowball Stemming
    for word in tokens:
        stm = snowBall_stemmer.stem(word.lower())
        stemmed_tokens.append(stm)
        
    # Sentence Reconstruction
    processed_sentence = " ".join(stemmed_tokens)
    return processed_sentence

# 3. Create Flask API POST /predict
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided in request'}), 400
        
    text = data['text']
    
    if not model_loaded:
        # Fallback dummy data so UI works before models are trained
        return jsonify({
            'prediction': 'Data Science (Model not loaded)',
            'confidence': 90,
            'top_matches': ['Data Science', 'Machine Learning', 'Software Engineer']
        })

    try:
        # Preprocess text
        preprocessed_text = preprocess_text(text)
        
        # Vectorizer Transform
        X = vectorizer.transform([preprocessed_text])
        
        # Model Predict
        prediction = model.predict(X)[0]
        
        # Extra: Calculate confidence if possible
        confidence = 94
        top_matches = [str(prediction)]
        try:
            probabilities = model.predict_proba(X)[0]
            top_4_indices = probabilities.argsort()[-4:][::-1]
            classes = model.classes_
            top_matches = [classes[i] for i in top_4_indices]
            confidence = int(max(probabilities) * 100)
        except:
            pass
            
        # Return JSON response
        return jsonify({
            'prediction': str(prediction),
            'confidence': confidence,
            'top_matches': top_matches
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
