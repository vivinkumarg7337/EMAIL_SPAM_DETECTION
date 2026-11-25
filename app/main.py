"""MailGuard - email spam detection served from an Atlas Engine trained model."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
REPORTS = ROOT / "reports"

STOPWORDS = set("""a about above after again against all am an and any are as at be because been before
being below between both but by can cannot could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with you your yours yourself
yourselves""".split())


def keep(term: object) -> bool:
    t = str(term)
    return len(t) >= 3 and t not in STOPWORDS and not t.isdigit()

app = FastAPI(title="MailGuard", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "app" / "templates")

_state: dict = {}


@app.on_event("startup")
def load_model():
    if not (MODEL_DIR / "pipeline.joblib").exists():
        raise RuntimeError("No trained model found. Run scripts/train_sweep.py first.")
    pipe = joblib.load(MODEL_DIR / "pipeline.joblib")
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    decision = MODEL_DIR / 'decision.json'
    _state['threshold'] = json.loads(decision.read_text())['threshold'] if decision.exists() else 0.5
    _state.update(pipe=pipe, meta=meta, classes=meta["classes"],
                  vec=pipe.named_steps["vec"], clf=pipe.named_steps["model"])


class Email(BaseModel):
    subject: str = ""
    body: str = ""


def as_text(subject: str, body: str) -> str:
    return f"{subject or ''}\n{body or ''}"


def explain(text: str, limit: int = 14):
    """Top n-grams by absolute contribution, tagged with the class they push toward."""
    vec, clf = _state["vec"], _state["clf"]
    row = vec.transform([text])
    names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    values = row.toarray()[0]
    idx = np.nonzero(values)[0]
    if len(idx) == 0:
        return []
    contrib = values[idx] * coef[idx]
    order = np.argsort(np.abs(contrib))[::-1]
    out = []
    for j in order:
        name = str(names[idx[j]])
        if not keep(name):
            continue
        out.append({"term": name, "weight": round(float(contrib[j]), 4),
                    "direction": "spam" if contrib[j] > 0 else "legitimate"})
        if len(out) >= limit:
            break
    return out


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/health")
def health():
    return {"ok": True, "model": _state["meta"].get("kind"), "classes": _state["classes"], "threshold": _state['threshold']}


@app.post("/api/predict")
def predict(email: Email):
    text = as_text(email.subject, email.body)
    if not text.strip():
        raise HTTPException(400, "Add a subject or a body first.")
    t0 = time.perf_counter()
    proba = _state["pipe"].predict_proba([text])[0]
    classes = _state["classes"]
    spam_p = float(proba[classes.index("1")]) if "1" in classes else float(proba[-1])
    spam = spam_p >= _state['threshold']
    return {
        "label": "spam" if spam else "legitimate",
        "spam_probability": spam_p,
        "threshold": _state['threshold'],
        "confidence": round(max(spam_p, 1 - spam_p), 4),
        "explanation": explain(text),
        "terms_in_text": int(_state["vec"].transform([text]).nnz),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.get("/api/examples")
def examples():
    path = ROOT / "data" / "test.csv"
    if not path.exists():
        return {"spam": [], "legitimate": []}
    df = pd.read_csv(path)
    out = {}
    for label, key in ((1, "spam"), (0, "legitimate")):
        part = df[df["label"] == label].head(3)
        out[key] = [{"subject": r.subject, "body": r.body} for r in part.itertuples()]
    return out


@app.get("/api/metrics")
def metrics():
    if not (REPORTS / "sweep.json").exists():
        raise HTTPException(404, "No sweep report yet.")
    sweep = json.loads((REPORTS / "sweep.json").read_text())
    terms = {}
    if (MODEL_DIR / "top_terms.json").exists():
        terms = json.loads((MODEL_DIR / "top_terms.json").read_text())
    return {"sweep": sweep, "terms": terms}


@app.post("/api/batch")
async def batch(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not read that CSV: {exc}") from exc
    if "text" in df.columns:
        texts = df["text"].fillna("").astype(str)
    elif "body" in df.columns:
        subject = df["subject"].fillna("").astype(str) if "subject" in df.columns else ""
        texts = subject + "\n" + df["body"].fillna("").astype(str)
    elif "subject" in df.columns:
        texts = df["subject"].fillna("").astype(str)
    else:
        raise HTTPException(400, "CSV needs a 'text' column, or 'subject' and/or 'body' columns.")

    proba = _state["pipe"].predict_proba(texts.tolist())
    classes = _state["classes"]
    spam_p = proba[:, classes.index("1")] if "1" in classes else proba[:, -1]
    out = df.copy()
    out["prediction"] = np.where(spam_p >= _state['threshold'], "spam", "legitimate")
    out["spam_probability"] = np.round(spam_p, 4)

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    name = (file.filename or "emails.csv").rsplit(".", 1)[0]
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{name}_predictions.csv"',
            "X-Rows": str(len(out)),
            "X-Spam": str(int((out["prediction"] == "spam").sum())),
        },
    )
