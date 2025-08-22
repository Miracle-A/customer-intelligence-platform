from flask import Flask, request, jsonify
import os, math, warnings
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ---------- Config ----------
DATA_LOCAL = os.getenv("DATA_LOCAL_PATH", "/mnt/data/customer_intelligence_dataset.csv")
DATA_GCS   = os.getenv("DATA_GCS_URI", "").strip()  # e.g. gs://bucket/file.csv
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Ensure VADER is present
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
app = Flask(__name__)

# ---------- Data helpers ----------
def load_csv() -> pd.DataFrame:
    if DATA_GCS:
        try:
            import gcsfs  # requires gcsfs in requirements
            df = pd.read_csv(DATA_GCS)
        except Exception as e:
            app.logger.warning(f"GCS read failed ({DATA_GCS}); falling back to local. {e}")
            df = pd.read_csv(DATA_LOCAL)
    else:
        df = pd.read_csv(DATA_LOCAL)

    for c in ["sale_date", "last_purchase_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "total_value" not in df.columns and {"price","quantity"}.issubset(df.columns):
        df["total_value"] = df["price"] * df["quantity"]

    if "sale_id" not in df.columns:
        df["sale_id"] = np.arange(1, len(df) + 1)

    return df

def build_customer_table(df: pd.DataFrame) -> pd.DataFrame:
    if "sale_date" not in df.columns:
        raise ValueError("Need 'sale_date' in dataset.")
    now_date = df["sale_date"].dropna().max()
    grp = df.groupby("customer_id", as_index=False).agg(
        first_purchase=("sale_date", "min"),
        last_purchase=("sale_date", "max"),
        frequency=("sale_id", "nunique"),
        monetary=("total_value", "sum"),
    )
    grp["recency_days"] = (now_date - grp["last_purchase"]).dt.days
    attrs = ["age","gender","region","tenure_months","churn","segment"]
    attrs_present = [c for c in attrs if c in df.columns]
    if attrs_present:
        stable = df.sort_values("sale_date").groupby("customer_id").agg({c:"last" for c in attrs_present}).reset_index()
        return grp.merge(stable, on="customer_id", how="left")
    return grp

# ---------- Modeling helpers ----------
def train_churn_model(customers: pd.DataFrame, save=True):
    if "churn" not in customers.columns:
        raise ValueError("No 'churn' column found.")
    df = customers.copy()
    y = df["churn"].astype(int)
    numeric_cols = [c for c in ["recency_days","frequency","monetary","tenure_months","age"] if c in df.columns]
    cat_cols     = [c for c in ["gender","region","segment"] if c in df.columns]
    X = df[numeric_cols + cat_cols].copy()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    model = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:,1]
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": float(metrics.accuracy_score(yte, pred)),
        "precision": float(metrics.precision_score(yte, pred, zero_division=0)),
        "recall": float(metrics.recall_score(yte, pred, zero_division=0)),
        "f1": float(metrics.f1_score(yte, pred, zero_division=0)),
        "f2": float(metrics.fbeta_score(yte, pred, beta=2, zero_division=0)),
        "roc_auc": float(metrics.roc_auc_score(yte, prob)),
        "pr_auc": float(metrics.average_precision_score(yte, prob)),
    }
    if save:
        joblib.dump(model, os.path.join(MODELS_DIR, "churn_model.pkl"))
        joblib.dump({"numeric_cols": numeric_cols, "cat_cols": cat_cols},
                    os.path.join(MODELS_DIR, "churn_meta.pkl"))
    return out

def forecast_linear_monthly(df: pd.DataFrame, horizon=6):
    ts = df.dropna(subset=["sale_date"]).copy()
    ts["YearMonth"] = ts["sale_date"].dt.to_period("M").astype(str)
    mrev = ts.groupby("YearMonth")["total_value"].sum().reset_index()
    mrev["t"] = np.arange(len(mrev))
    X, y = mrev[["t"]], mrev["total_value"].values
    if len(mrev) > (horizon + 3):
        split = len(mrev) - horizon
        Xtr, ytr = X.iloc[:split], y[:split]
        Xte, yte = X.iloc[split:], y[split:]
    else:
        Xtr, ytr = X, y
        Xte, yte = X.iloc[0:0], y[0:0]
    lr = LinearRegression().fit(Xtr, ytr)
    last_t = int(X["t"].max())
    fut_t = np.arange(last_t+1, last_t+1+horizon)
    yhat_fut = lr.predict(pd.DataFrame({"t": fut_t}))
    result = {
        "history": [{"period": p, "value": float(v)} for p, v in zip(mrev["YearMonth"], y)],
        "forecast": [{"period": str(pd.Period(mrev["YearMonth"].iloc[-1], freq="M")+i), "value": float(v)} for i, v in enumerate(yhat_fut, start=1)],
        "horizon": horizon,
        "model": "LinearRegression"
    }
    if len(Xte) > 0:
        yhat = lr.predict(Xte)
        mae  = metrics.mean_absolute_error(yte, yhat)
        rmse = math.sqrt(metrics.mean_squared_error(yte, yhat))
        mape = float(np.mean(np.abs((yte - yhat)/np.maximum(1e-9, yte))) * 100)
        result["metrics"] = {"mae": float(mae), "rmse": float(rmse), "mape": mape}
    return result

def next_best_action(churn_prob: float, monetary: float = 0.0, segment: Optional[str] = None, recency_days: Optional[float] = None) -> str:
    seg = (segment or "Unknown").lower()
    r = recency_days if recency_days is not None else 999
    if churn_prob >= 0.6:
        return "VIP Retention: 15% loyalty + outreach" if monetary >= 500 else "Save Offer: 10% off + reminders"
    if churn_prob >= 0.4:
        return "Reactivation: win-back + small coupon" if seg.startswith("new") or r > 45 else "Engagement: product recs + content drip"
    return "Upsell: bundle/premium" if monetary >= 1000 else "Nurture: newsletter + light promo"

# ---------- Routes ----------
@app.route("/health", methods=["GET"])
def health(): return jsonify({"status": "ok"}), 200

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Customer Intelligence Platform API",
        "endpoints": {
            "GET  /health": "Service health",
            "POST /train": "Train churn model",
            "POST /predict/churn": "Score customers",
            "GET  /forecast/sales?horizon=6": "Sales forecast",
            "POST /sentiment": "VADER for texts or dataset feedback",
            "POST /segment/kmeans": "RFM K-Means",
            "POST /prescriptions": "Next Best Action"
        },
        "env": {"DATA_LOCAL_PATH": DATA_LOCAL, "DATA_GCS_URI": DATA_GCS}
    }), 200

@app.route("/train", methods=["POST"])
def train():
    try:
        df = load_csv()
        customers = build_customer_table(df)
        metrics_out = train_churn_model(customers, save=True)
        return jsonify({"message": "trained", "metrics": metrics_out}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/churn", methods=["POST"])
def predict_churn():
    try:
        model_p = os.path.join(MODELS_DIR, "churn_model.pkl")
        meta_p  = os.path.join(MODELS_DIR, "churn_meta.pkl")
        if not os.path.exists(model_p):
            df = load_csv(); customers = build_customer_table(df); train_churn_model(customers, save=True)
        model = joblib.load(model_p)
        meta  = joblib.load(meta_p)
        num_cols, cat_cols = meta["numeric_cols"], meta["cat_cols"]

        payload = request.get_json(silent=True) or {}
        thr = float(payload.get("threshold", 0.5))

        if "customers" in payload:
            X = pd.DataFrame(payload["customers"])
        else:
            df = load_csv()
            X = build_customer_table(df)
            if "customer_id" in X.columns:
                pass

        for c in num_cols + cat_cols:
            if c not in X.columns:
                X[c] = np.nan

        out = X.copy()
        probs = model.predict_proba(X[num_cols + cat_cols])[:,1]
        out["churn_prob"] = probs
        out["churn_label"] = (probs >= thr).astype(int)

        if "monetary" in out.columns and "recency_days" in out.columns:
            out["next_best_action"] = [
                next_best_action(float(p), float(out.loc[i].get("monetary", 0)),
                                 str(out.loc[i].get("segment", "")),
                                 float(out.loc[i].get("recency_days", 999)))
                for i, p in enumerate(probs)
            ]
        return jsonify({"count": int(len(out)), "threshold": thr, "results": out.to_dict(orient="records")}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/forecast/sales", methods=["GET"])
def forecast_sales():
    try:
        horizon = int(request.args.get("horizon", 6))
        df = load_csv()
        return jsonify(forecast_linear_monthly(df, horizon)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/sentiment", methods=["POST"])
def sentiment():
    try:
        payload = request.get_json(silent=True) or {}
        if "texts" in payload and isinstance(payload["texts"], list):
            texts = [str(x) for x in payload["texts"]]
        else:
            df = load_csv()
            texts = [str(x) for x in df.get("feedback_text", pd.Series(dtype=str)).dropna().tolist()]
        results = []
        for t in texts:
            s = sia.polarity_scores(t)
            comp = s["compound"]
            label = "Positive" if comp >= 0.05 else ("Negative" if comp <= -0.05 else "Neutral")
            results.append({"text": t, "compound": comp, "label": label})
        return jsonify({"count": len(results), "results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/segment/kmeans", methods=["POST"])
def segment_kmeans():
    try:
        payload = request.get_json(silent=True) or {}
        k = int(payload.get("k", 4))
        df = load_csv()
        cust = build_customer_table(df)
        cols = ["recency_days","frequency","monetary"]
        if "tenure_months" in cust.columns: cols.append("tenure_months")
        use = cust.dropna(subset=cols).copy()
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        use["cluster"] = km.fit_predict(use[cols].values)
        prof = use.groupby("cluster")[cols].mean().round(2).reset_index()
        return jsonify({"k": k,
                        "counts": use["cluster"].value_counts().sort_index().to_dict(),
                        "profiles": prof.to_dict(orient="records")}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/prescriptions", methods=["POST"])
def prescriptions():
    try:
        payload = request.get_json(silent=True) or {}
        items = payload.get("customers", [])
        out = []
        for row in items:
            action = next_best_action(float(row.get("churn_prob", 0.0)),
                                      float(row.get("monetary", 0.0)),
                                      str(row.get("segment", "")),
                                      float(row.get("recency_days", 999)))
            r = dict(row); r["next_best_action"] = action
            out.append(r)
        return jsonify({"count": len(out), "results": out}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8501")))
