import ssl
import pandas as pd
import nltk
import string
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Fix SSL certificate issue on macOS before downloading NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("reviews.csv")

# Stopwords
stop_words = set(stopwords.words('english'))


# Function for cleaning text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


# Apply cleaning
df['cleaned_review'] = df['review'].apply(clean_text)


# Generate labels using TextBlob (used to train the ML model)
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


df['sentiment'] = df['cleaned_review'].apply(get_sentiment)

# Print labelled data
print("\n--- Labelled Reviews ---")
print(df[['review', 'cleaned_review', 'sentiment']])

# --- NLP Model: TF-IDF + Logistic Regression ---

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# Train/test split (only if enough samples exist)
if len(df) >= 4:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))
else:
    # Train on full data when dataset is too small to split
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    print("\n--- Model trained on full dataset (too small to split) ---")

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nModel and vectorizer saved: sentiment_model.pkl, tfidf_vectorizer.pkl")


# Prediction helper used by the API
def predict_sentiment(review_text: str) -> dict:
    cleaned = clean_text(review_text)
    vector = vectorizer.transform([cleaned])
    label = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = round(float(max(proba)), 4)
    return {
        "review": review_text,
        "cleaned_review": cleaned,
        "sentiment": label,
        "confidence": confidence,
    }
