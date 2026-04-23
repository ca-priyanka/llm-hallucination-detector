"""
LLM Hallucination Guardrail API
================================
POST /ask      – ask a question, get LLM answer + hallucination score
POST /score    – score a pre-existing question + answer pair
GET  /health   – liveness check
GET  /metrics  – last evaluation metrics
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

from features.extractor import extract_features, features_to_array

# ── Load model & metadata ─────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent
MODEL_PATH = BASE / "models" / "hallucination_model.joblib"
META_PATH  = BASE / "models" / "model_meta.json"

_model = None
_meta  = {}

def _load_model():
    global _model, _meta
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            _meta = json.loads(META_PATH.read_text())
    else:
        _model = None  # Will use heuristic fallback


def _heuristic_score(features: dict) -> float:
    """
    Rule-based fallback when no trained model is available.
    Returns a 0–100 risk score.
    """
    w = {
        "semantic_sim":        -40,   # high similarity → low risk
        "confidence_score":     25,   # overconfidence → higher risk
        "ner_mismatch":         20,   # entity mismatch → higher risk
        "contradiction_score":  30,   # contradiction → higher risk
        "numeric_mismatch":     15,   # number mismatch → higher risk
    }
    score = 50.0  # baseline
    score += w["semantic_sim"]        * features["semantic_sim"]
    score += w["confidence_score"]    * features["confidence_score"]
    score += w["ner_mismatch"]        * features["ner_mismatch"]
    score += w["contradiction_score"] * features["contradiction_score"]
    score += w["numeric_mismatch"]    * features["numeric_mismatch"]
    return float(np.clip(score, 0, 100))


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Hallucination Guardrail API",
    description="Scores LLM responses for hallucination risk using retrieval + ML.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    _load_model()


# ── Schemas ───────────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    question: str = Field(..., min_length=5, example="When was the Eiffel Tower built?")
    answer: str   = Field(..., min_length=5, example="The Eiffel Tower was built in 1902.")
    grounding: str | None = Field(
        None,
        description="Optional pre-fetched grounding text. If omitted, Wikipedia is queried.",
    )


class FeatureBreakdown(BaseModel):
    semantic_similarity: float
    confidence_word_density: float
    ner_mismatch_ratio: float
    contradiction_probability: float
    numeric_mismatch_ratio: float


class ScoreResponse(BaseModel):
    hallucination_risk: int          # 0–100
    risk_label: str                  # LOW / MEDIUM / HIGH
    safe_to_show: bool
    confidence_score: float          # model probability 0–1
    grounding_source: str            # snippet used for verification
    features: FeatureBreakdown
    model_used: str
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

# ── /ask schemas ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, example="When was the Eiffel Tower built?")
    system_prompt: str | None = Field(
        None,
        description="Optional system prompt sent to the LLM.",
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    llm_model: str
    hallucination_risk: int
    risk_label: str
    safe_to_show: bool
    confidence_score: float
    grounding_source: str
    features: FeatureBreakdown
    guardrail_model: str
    total_latency_ms: float
    llm_latency_ms: float
    guardrail_latency_ms: float


# ── /ask endpoint ─────────────────────────────────────────────────────────────

def _call_llm(question: str, system_prompt: str | None) -> tuple[str, float, str]:
    """Call Claude (Anthropic) and return (answer, latency_ms, model_name)."""
    try:
        import anthropic
    except ImportError:
        raise HTTPException(status_code=500, detail="anthropic package not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="ANTHROPIC_API_KEY not set. Add it to your .env file.",
        )

    client = anthropic.Anthropic(api_key=api_key)
    model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")  # fast + cheap by default

    system = system_prompt or "You are a helpful assistant. Answer factual questions concisely and accurately."

    t0 = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    llm_ms = round((time.perf_counter() - t0) * 1000, 2)

    answer = response.content[0].text.strip()
    return answer, llm_ms, model


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t_total = time.perf_counter()

    # 1. Call LLM
    answer, llm_ms, llm_model = _call_llm(req.question, req.system_prompt)

    # 2. Score the response
    t_guard = time.perf_counter()
    feat_dict = extract_features(req.question, answer)
    feat_array = features_to_array(feat_dict).reshape(1, -1)

    if _model is not None:
        proba = float(_model.predict_proba(feat_array)[0][1])
        threshold = _meta.get("best_threshold", 0.5)
        risk_score = int(round(proba * 100))
        guardrail_model = _meta.get("model_name", "trained_model")
    else:
        risk_score = int(_heuristic_score(feat_dict))
        proba = risk_score / 100.0
        threshold = 0.5
        guardrail_model = "heuristic_fallback"

    if risk_score < 30:
        risk_label = "LOW"
    elif risk_score < 65:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    guard_ms = round((time.perf_counter() - t_guard) * 1000, 2)
    total_ms = round((time.perf_counter() - t_total) * 1000, 2)

    return AskResponse(
        question=req.question,
        answer=answer,
        llm_model=llm_model,
        hallucination_risk=risk_score,
        risk_label=risk_label,
        safe_to_show=proba < threshold,
        confidence_score=round(proba, 4),
        grounding_source=feat_dict["_grounding_snippet"],
        features=FeatureBreakdown(
            semantic_similarity=feat_dict["semantic_sim"],
            confidence_word_density=feat_dict["confidence_score"],
            ner_mismatch_ratio=feat_dict["ner_mismatch"],
            contradiction_probability=feat_dict["contradiction_score"],
            numeric_mismatch_ratio=feat_dict["numeric_mismatch"],
        ),
        guardrail_model=guardrail_model,
        total_latency_ms=total_ms,
        llm_latency_ms=llm_ms,
        guardrail_latency_ms=guard_ms,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_name": _meta.get("model_name", "heuristic"),
    }


@app.get("/metrics")
def get_metrics():
    metrics_path = BASE / "reports" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="No metrics found. Run train.py first.")
    return json.loads(metrics_path.read_text())


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    t0 = time.perf_counter()

    # Extract features
    feat_dict = extract_features(req.question, req.answer, grounding=req.grounding)
    feat_array = features_to_array(feat_dict).reshape(1, -1)

    # Score
    if _model is not None:
        proba = float(_model.predict_proba(feat_array)[0][1])
        threshold = _meta.get("best_threshold", 0.5)
        risk_score = int(round(proba * 100))
        model_used = _meta.get("model_name", "trained_model")
    else:
        risk_score = int(_heuristic_score(feat_dict))
        proba = risk_score / 100.0
        threshold = 0.5
        model_used = "heuristic_fallback"

    # Labels
    if risk_score < 30:
        risk_label = "LOW"
    elif risk_score < 65:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    safe_to_show = proba < threshold

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return ScoreResponse(
        hallucination_risk=risk_score,
        risk_label=risk_label,
        safe_to_show=safe_to_show,
        confidence_score=round(proba, 4),
        grounding_source=feat_dict["_grounding_snippet"],
        features=FeatureBreakdown(
            semantic_similarity=feat_dict["semantic_sim"],
            confidence_word_density=feat_dict["confidence_score"],
            ner_mismatch_ratio=feat_dict["ner_mismatch"],
            contradiction_probability=feat_dict["contradiction_score"],
            numeric_mismatch_ratio=feat_dict["numeric_mismatch"],
        ),
        model_used=model_used,
        latency_ms=latency_ms,
    )
