import ssl
import string
import os
import joblib
import nltk
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Fix SSL certificate issue on macOS for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# Load saved model and vectorizer
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise RuntimeError("Model files not found. Run app.py first to train and save the model.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Load reviews CSV for the /reviews and /sentiment-summary endpoints
df = pd.read_csv("reviews.csv")


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in stop_words)


app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    review: str


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


@app.post("/predict")
def predict(request: ReviewRequest):
    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    cleaned = clean_text(request.review)
    vector = vectorizer.transform([cleaned])
    label = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = round(float(max(proba)), 4)
    return {
        "review": request.review,
        "cleaned_review": cleaned,
        "sentiment": label,
        "confidence": confidence,
    }


@app.get("/reviews")
def get_all_reviews():
    df['cleaned_review'] = df['review'].apply(clean_text)
    df['sentiment'] = df['cleaned_review'].apply(
        lambda t: model.predict(vectorizer.transform([t]))[0]
    )
    records = df[['review', 'cleaned_review', 'sentiment']].to_dict(orient='records')
    return {"total": len(records), "reviews": records}


@app.get("/sentiment-summary")
def sentiment_summary():
    df['cleaned_review'] = df['review'].apply(clean_text)
    df['sentiment'] = df['cleaned_review'].apply(
        lambda t: model.predict(vectorizer.transform([t]))[0]
    )
    counts = df['sentiment'].value_counts().to_dict()
    return {"summary": counts}
