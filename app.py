# app.py
from flask import Flask, request, jsonify, send_file
import os, json, logging
import numpy as np
from datetime import datetime

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# Sentiment (no external service needed)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -------------------- App & Logging --------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Paths & Defaults -----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Default to <repo>/models; can override with env vars
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))

CHURN_MODEL_PATH = os.getenv(
    "CHURN_MODEL_PATH",
    os.path.join(MODELS_DIR, "rf_churn_model.joblib")
)
FORECAST_MODEL_PATH = os.getenv(
    "FORECAST_MODEL_PATH",
    os.path.join(MODELS_DIR, "best_forecast_model_Ridge_prod.joblib")
)

# Unified /predict models (for the exercise)
LOGREG_MODEL_PATH = os.getenv(
    "LOGREG_MODEL_PATH",
    os.path.join(MODELS_DIR, "logreg_model.joblib")
)
KMEANS_MODEL_PATH = os.getenv(
    "KMEANS_MODEL_PATH",
    os.path.join(MODELS_DIR, "kmeans_model.joblib")
)
PREDICT_DATA_PATH = os.getenv(
    "PREDICT_DATA_PATH",
    os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_transactions.csv")
)
# Feature contract for unified /predict
PREDICT_FEATURES = [
    "age",
    "tenure_months",
    "num_purchases",
    "avg_spent",
]

# Threshold JSON (prefer rf_churn_threshold.json if present)
CHURN_THRESH_PATH = os.getenv(
    "CHURN_THRESH_PATH",
    os.path.join(MODELS_DIR, "rf_threshold.json")  # fallback
)
alt_thresh = os.path.join(MODELS_DIR, "rf_churn_threshold.json")
if not os.path.exists(CHURN_THRESH_PATH) and os.path.exists(alt_thresh):
    CHURN_THRESH_PATH = alt_thresh

# Static “serve-only” assets (optional)
INSIGHTS_FILE = os.getenv("INSIGHTS_FILE", os.path.join(PROJECT_ROOT, "customer_insights_mistral.txt"))
AUDIO_FILE    = os.getenv("AUDIO_FILE",    os.path.join(PROJECT_ROOT, "audio_output", "insights_from_file.mp3"))

# -------------------- NLTK (VADER) ---------------------
def _ensure_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
_ensure_vader()
_vader = SentimentIntensityAnalyzer()

# -------------------- Utils ----------------------------
def _file_mtime(path: str):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    except Exception:
        return None

def _load_joblib(path: str, name: str):
    if not os.path.exists(path):
        logger.warning("%s not found at %s", name, path)
        return None
    try:
        obj = joblib.load(path)
        logger.info("Loaded %s from %s", name, path)
        return obj
    except Exception as e:
        logger.exception("Failed loading %s: %s", name, e)
        return None

def _load_threshold_json(path: str, key: str = "best_threshold_f2", default: float = 0.50) -> float:
    if not os.path.exists(path):
        logger.warning("Threshold file not found at %s; defaulting to %.2f", path, default)
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        th = float(data.get(key, default))
        logger.info("Loaded threshold %.3f from %s", th, path)
        return th
    except Exception as e:
        logger.exception("Failed reading threshold json: %s; defaulting to %.2f", e, default)
        return default

def _ensure_feature_array(features):
    if features is None:
        raise ValueError("Missing 'features' in request body.")
    return np.array(features, dtype=float).reshape(1, -1)

def _build_training_frame(path: str) -> pd.DataFrame | None:
    """Aggregate per-customer features for simple training of logreg/kmeans."""
    if not os.path.exists(path):
        logger.warning("Training data not found at %s", path)
        return None
    try:
        df = pd.read_csv(path)
        # Parse dates if present
        for c in ("sale_date", "last_purchase_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        # Compute per-customer aggregates
        if {"customer_id", "sale_id"}.issubset(df.columns):
            grp = df.groupby("customer_id", as_index=False).agg(
                num_purchases=("sale_id", "count"),
                total_spent=("total_value", "sum"),
                last_purchase=("sale_date", "max"),
            )
        else:
            # Fallback if no IDs; operate on whole frame
            grp = pd.DataFrame({
                "num_purchases": [int(len(df))],
                "total_spent": [float(df.get("total_value", pd.Series([0])).sum())],
                "last_purchase": [pd.to_datetime(df.get("sale_date", pd.Series([None]))).max()],
            })

        # Bring stable attributes if present (last known per customer)
        attrs = ["age", "tenure_months", "churn"]
        attrs_present = [c for c in attrs if c in df.columns]
        if attrs_present and "customer_id" in df.columns:
            stable = df.sort_values(by=df.columns.tolist()).groupby("customer_id").agg({c: "last" for c in attrs_present}).reset_index()
            grp = grp.merge(stable, on="customer_id", how="left")
        # Derive recency days
        now_date = df.get("sale_date")
        if now_date is not None and not df["sale_date"].dropna().empty:
            now_d = df["sale_date"].dropna().max()
            grp["recency_days"] = (now_d - grp["last_purchase"]).dt.days
        # Final features
        grp["avg_spent"] = grp["total_spent"] / grp["num_purchases"].replace({0: np.nan})
        grp["avg_spent"].fillna(0, inplace=True)
        # Keep only required columns
        for c in PREDICT_FEATURES:
            if c not in grp.columns:
                grp[c] = 0.0
        cols = PREDICT_FEATURES + (["churn"] if "churn" in grp.columns else [])
        return grp[cols]
    except Exception as e:
        logger.exception("Failed building training frame: %s", e)
        return None

def _train_unified_models_if_missing():
    """Train logreg and kmeans if their joblib files are missing, using local processed data."""
    need_logreg = not os.path.exists(LOGREG_MODEL_PATH)
    need_kmeans = not os.path.exists(KMEANS_MODEL_PATH)
    if not (need_logreg or need_kmeans):
        return
    df = _build_training_frame(PREDICT_DATA_PATH)
    if df is None or df.empty:
        logger.warning("Cannot train models: no data available at %s", PREDICT_DATA_PATH)
        return
    X = df[PREDICT_FEATURES].fillna(0).astype(float)
    # Train KMeans
    if need_kmeans:
        try:
            km = KMeans(n_clusters=4, n_init=10, random_state=42)
            km.fit(X)
            joblib.dump(km, KMEANS_MODEL_PATH)
            logger.info("Saved KMeans to %s", KMEANS_MODEL_PATH)
        except Exception as e:
            logger.exception("KMeans training failed: %s", e)
    # Train Logistic Regression if label available
    if need_logreg and "churn" in df.columns:
        try:
            y = df["churn"].astype(int)
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            lr.fit(X, y)
            joblib.dump(lr, LOGREG_MODEL_PATH)
            logger.info("Saved LogisticRegression to %s", LOGREG_MODEL_PATH)
        except Exception as e:
            logger.exception("LogisticRegression training failed: %s", e)

# -------------------- Load Models on Startup ------------
churn_model     = _load_joblib(CHURN_MODEL_PATH,     "Random Forest Churn Model")
forecast_model  = _load_joblib(FORECAST_MODEL_PATH,  "Ridge Sales Forecast Model")
churn_threshold = _load_threshold_json(CHURN_THRESH_PATH, key="best_threshold_f2", default=0.50)

# Unified /predict models
_train_unified_models_if_missing()
predict_logreg_model = _load_joblib(LOGREG_MODEL_PATH, "Unified LogReg Model")
predict_kmeans_model = _load_joblib(KMEANS_MODEL_PATH, "Unified KMeans Model")

# -------------------- Helpers --------------------------
def _predict_churn_prob(arr: np.ndarray) -> float:
    """Return probability of churn (class 1). Pylance-safe with runtime guard."""
    if churn_model is None:
        raise RuntimeError(f"Churn model not loaded (expected at {CHURN_MODEL_PATH}).")
    if hasattr(churn_model, "predict_proba"):
        return float(churn_model.predict_proba(arr)[0, 1])
    if hasattr(churn_model, "decision_function"):
        score = churn_model.decision_function(arr).astype(float)
        return float((1.0 / (1.0 + np.exp(-score))).ravel()[0])
    # Fallback to hard label as prob {0,1}
    return float(int(churn_model.predict(arr)[0]))

# -------------------- Routes ---------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Customer Intelligence Platform API is running.",
        "endpoints": [
            "GET  /status",
            "GET  /predict/schema",
            "GET  /predict/churn/schema",
            "POST /predict",
            "POST /predict/churn",
            "POST /predict/forecast",
            "POST /sentiment",
            "POST /llm_insights",
            "GET/POST /tts_insights"
        ]
    })

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "models": {
            "churn_rf": {
                "loaded": churn_model is not None,
                "path": CHURN_MODEL_PATH,
                "mtime": _file_mtime(CHURN_MODEL_PATH),
                "threshold_f2": churn_threshold
            },
            "forecast_ridge": {
                "loaded": forecast_model is not None,
                "path": FORECAST_MODEL_PATH,
                "mtime": _file_mtime(FORECAST_MODEL_PATH)
            }
        },
        "assets": {
            "insights_file": {"exists": os.path.exists(INSIGHTS_FILE), "path": INSIGHTS_FILE, "mtime": _file_mtime(INSIGHTS_FILE)},
            "audio_file":    {"exists": os.path.exists(AUDIO_FILE),    "path": AUDIO_FILE,    "mtime": _file_mtime(AUDIO_FILE)}
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/__routes__", methods=["GET"])
def list_routes():
    """List all registered routes (for debugging)."""
    rules = []
    for r in app.url_map.iter_rules():
        m = set(r.methods) if r.methods else set()
        m = m - {"HEAD", "OPTIONS"}
        rules.append({
            "rule": str(r),
            "endpoint": r.endpoint,
            "methods": sorted(list(m))
        })
    return jsonify({"routes": sorted(rules, key=lambda x: x["rule"])})

@app.route("/predict/churn", methods=["POST"])
def predict_churn():
    """
    Body:
    { "features": [ ... ] }  # exact order your RF expects
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
        arr = _ensure_feature_array(payload.get("features"))
        proba = _predict_churn_prob(arr)
        label = int(proba >= churn_threshold)
        return jsonify({
            "model": "rf_churn",
            "threshold_f2": churn_threshold,
            "features_len": int(arr.shape[1]),
            "prediction": {"prob_churn": proba, "label": label}
        })
    except Exception as e:
        logger.exception("Churn prediction failed: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/predict/churn/schema", methods=["GET"])
def predict_churn_schema():
    """Return the expected feature order and size for the churn model."""
    if churn_model is None:
        return jsonify({"error": f"Churn model not loaded from {CHURN_MODEL_PATH}"}), 500
    try:
        n = int(getattr(churn_model, "n_features_in_", 0) or 0)
        names = getattr(churn_model, "feature_names_in_", None)
        feature_names = [str(x) for x in list(names)] if names is not None else None
        return jsonify({
            "model": "rf_churn",
            "n_features": n,
            "feature_names": feature_names,
            "threshold_f2": churn_threshold,
            "example": {"features": [0] * n},
            "notes": "Send POST /predict/churn with JSON { 'features': [..] } in this exact order."
        })
    except Exception as e:
        logger.exception("Schema retrieval failed: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/predict/schema", methods=["GET"])
def unified_predict_schema():
    """Schema for the unified /predict endpoint (model_type in {logreg,kmeans})."""
    models = {
        "logreg": {"loaded": predict_logreg_model is not None},
        "kmeans": {"loaded": predict_kmeans_model is not None},
    }
    return jsonify({
        "models": models,
        "n_features": len(PREDICT_FEATURES),
        "feature_names": PREDICT_FEATURES,
        "example": {"model_type": "logreg", "features": [0] * len(PREDICT_FEATURES)}
    })

@app.route("/predict", methods=["POST"])
def unified_predict():
    """Unified prediction endpoint.
    Body: { "model_type": "logreg"|"kmeans", "features": [..] }
    """
    payload = request.get_json(force=True, silent=True) or {}
    model_type = str(payload.get("model_type", "")).lower().strip()
    feats = payload.get("features")
    arr = _ensure_feature_array(feats)
    if int(arr.shape[1]) != len(PREDICT_FEATURES):
        return jsonify({
            "error": f"Expected {len(PREDICT_FEATURES)} features ({PREDICT_FEATURES}), got {int(arr.shape[1])}."
        }), 400
    if model_type == "logreg":
        if predict_logreg_model is None:
            return jsonify({"error": f"LogReg model not available at {LOGREG_MODEL_PATH}"}), 500
        try:
            proba = None
            if hasattr(predict_logreg_model, "predict_proba"):
                proba = float(predict_logreg_model.predict_proba(arr)[0, 1])
            label = int(predict_logreg_model.predict(arr)[0])
            return jsonify({
                "model": "logreg",
                "features_len": int(arr.shape[1]),
                "prediction": {"label": label, **({"prob_churn": proba} if proba is not None else {})}
            })
        except Exception as e:
            logger.exception("Unified logreg prediction failed: %s", e)
            return jsonify({"error": str(e)}), 500
    elif model_type == "kmeans":
        if predict_kmeans_model is None:
            return jsonify({"error": f"KMeans model not available at {KMEANS_MODEL_PATH}"}), 500
        try:
            label = int(predict_kmeans_model.predict(arr)[0])
            return jsonify({
                "model": "kmeans",
                "features_len": int(arr.shape[1]),
                "prediction": {"cluster": label}
            })
        except Exception as e:
            logger.exception("Unified kmeans prediction failed: %s", e)
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid or missing 'model_type'. Use 'logreg' or 'kmeans'."}), 400

@app.route("/predict/forecast", methods=["POST"])
def predict_forecast():
    """
    Body:
    { "features": [ ... ] }  # exact order your Ridge model expects
    """
    if forecast_model is None:
        return jsonify({"error": f"Forecast model not loaded from {FORECAST_MODEL_PATH}"}), 500
    try:
        payload = request.get_json(force=True, silent=True) or {}
        arr = _ensure_feature_array(payload.get("features"))
        y_hat = float(forecast_model.predict(arr)[0])
        return jsonify({
            "model": "ridge_forecast",
            "features_len": int(arr.shape[1]),
            "prediction": y_hat
        })
    except Exception as e:
        logger.exception("Forecast prediction failed: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/sentiment", methods=["POST"])
def sentiment():
    """Body: { "text": "I love this product!" }"""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        text = payload.get("text", "")
        if not text:
            return jsonify({"error": "Please provide 'text' in the request body."}), 400
        return jsonify({"text": text, "scores": _vader.polarity_scores(text)})
    except Exception as e:
        logger.exception("Sentiment failed: %s", e)
        return jsonify({"error": str(e)}), 500

