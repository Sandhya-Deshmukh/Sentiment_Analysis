import { useState, useEffect } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

function Badge({ sentiment }) {
  const colors = {
    Positive: "#22c55e",
    Negative: "#ef4444",
    Neutral: "#f59e0b",
  };
  return (
    <span
      style={{
        background: colors[sentiment] || "#6b7280",
        color: "#fff",
        borderRadius: "999px",
        padding: "3px 14px",
        fontWeight: 600,
        fontSize: "0.82rem",
      }}
    >
      {sentiment}
    </span>
  );
}

export default function App() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [reviews, setReviews] = useState([]);
  const [activeTab, setActiveTab] = useState("predict");

  useEffect(() => {
    fetchReviews();
  }, []);

  async function fetchReviews() {
    try {
      const res = await fetch(`${API}/reviews`);
      const data = await res.json();
      setReviews(data.reviews);
    } catch {
      console.error("Failed to fetch reviews");
    }
  }

  // Compute summary directly from reviews data
  const summary = reviews.reduce((acc, r) => {
    acc[r.sentiment] = (acc[r.sentiment] || 0) + 1;
    return acc;
  }, {});

  async function handlePredict() {
    if (!review.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review }),
      });
      const data = await res.json();
      setResult(data);
    } catch {
      setError("Could not connect to the API. Make sure the server is running.");
    }
    setLoading(false);
  }

  const total = reviews.length;
  const summaryItems = [
    { label: "Positive", color: "#22c55e", bg: "#f0fdf4" },
    { label: "Negative", color: "#ef4444", bg: "#fef2f2" },
    { label: "Neutral",  color: "#f59e0b", bg: "#fffbeb" },
    { label: "Total",    color: "#6366f1", bg: "#f5f3ff" },
  ];

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1>Sentiment Analysis</h1>
        <p>Analyze the sentiment of app reviews using NLP</p>
      </header>

      {/* Summary Cards */}
      <div className="cards">
        {summaryItems.map(({ label, color, bg }) => (
          <div className="card" key={label} style={{ background: bg, borderTop: `4px solid ${color}` }}>
            <div className="card-count" style={{ color }}>{label === "Total" ? total : (summary[label] || 0)}</div>
            <div className="card-label">{label} Reviews</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="tabs">
        <button className={`tab ${activeTab === "predict" ? "active" : ""}`} onClick={() => setActiveTab("predict")}>
          Predict Sentiment
        </button>
        <button className={`tab ${activeTab === "reviews" ? "active" : ""}`} onClick={() => setActiveTab("reviews")}>
          All Reviews ({reviews.length})
        </button>
      </div>

      {/* Predict Tab */}
      {activeTab === "predict" && (
        <div className="section">
          <label className="input-label">Enter a Review</label>
          <textarea
            rows={4}
            placeholder="e.g. The app is very slow and keeps crashing..."
            value={review}
            onChange={(e) => setReview(e.target.value)}
          />
          <button className="btn" onClick={handlePredict} disabled={loading || !review.trim()}>
            {loading ? "Analyzing..." : "Analyze Sentiment"}
          </button>

          {error && <p className="error">{error}</p>}

          {result && (
            <div className="result-box">
              <h3>Result</h3>
              <div className="result-row">
                <span className="result-label">Sentiment</span>
                <Badge sentiment={result.sentiment} />
              </div>
              <div className="result-row">
                <span className="result-label">Confidence</span>
                <span className="result-value">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="conf-bar-bg">
                <div className="conf-bar-fill" style={{ width: `${result.confidence * 100}%` }} />
              </div>
              <div className="result-row" style={{ marginTop: "12px" }}>
                <span className="result-label">Original</span>
                <span className="result-value">{result.review}</span>
              </div>
              <div className="result-row">
                <span className="result-label">Cleaned</span>
                <span className="result-value cleaned">{result.cleaned_review}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* All Reviews Tab */}
      {activeTab === "reviews" && (
        <div className="section">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Review</th>
                  <th>Cleaned Text</th>
                  <th>Sentiment</th>
                </tr>
              </thead>
              <tbody>
                {reviews.map((r, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{r.review}</td>
                    <td className="cleaned">{r.cleaned_review}</td>
                    <td><Badge sentiment={r.sentiment} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
