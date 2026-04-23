"""
Feature extraction pipeline for hallucination detection.

Features computed per (question, answer) pair:
  1. semantic_sim        – cosine similarity between answer and Wikipedia summary
  2. confidence_score    – density of overconfident hedge words
  3. ner_mismatch        – fraction of answer entities not found in grounding text
  4. contradiction_score – NLI contradiction probability (answer vs. grounding)
  5. numeric_mismatch    – numbers in answer not present in grounding text
"""

import re
import math
import unicodedata
from typing import Optional

import numpy as np

# ── lazy imports (kept at module level to allow mocking in tests) ─────────────
_sentence_model = None
_nli_pipeline = None
_nlp = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        _nli_pipeline = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1,
        )
    return _nli_pipeline


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── Wikipedia retrieval ───────────────────────────────────────────────────────

def fetch_wikipedia_summary(query: str, sentences: int = 5) -> Optional[str]:
    """Return the first `sentences` sentences of the top Wikipedia hit."""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        results = wikipedia.search(query, results=3)
        if not results:
            return None
        page = wikipedia.summary(results[0], sentences=sentences, auto_suggest=False)
        return page
    except Exception:
        return None


# ── Individual feature functions ─────────────────────────────────────────────

CONFIDENCE_WORDS = {
    "definitely", "always", "never", "certainly", "absolutely",
    "undoubtedly", "unquestionably", "obviously", "clearly", "proven",
    "guaranteed", "without doubt", "100%", "exactly", "precisely",
}


def feature_semantic_similarity(answer: str, grounding: Optional[str]) -> float:
    """Cosine similarity between answer and grounding embeddings. 0 if no grounding."""
    if not grounding:
        return 0.0
    model = _get_sentence_model()
    embs = model.encode([answer, grounding], convert_to_numpy=True)
    a, b = embs[0], embs[1]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return round(cos, 4)


def feature_confidence_score(answer: str) -> float:
    """Fraction of words that are overconfident hedge terms (0–1)."""
    text = answer.lower()
    words = text.split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in CONFIDENCE_WORDS)
    # also check bigrams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    hits += sum(1 for bg in bigrams if bg in CONFIDENCE_WORDS)
    return round(min(hits / len(words), 1.0), 4)


def feature_ner_mismatch(answer: str, grounding: Optional[str]) -> float:
    """Fraction of named entities in answer that are absent from grounding text."""
    if not grounding:
        return 0.5  # unknown — neutral penalty
    nlp = _get_nlp()
    answer_doc = nlp(answer)
    answer_ents = {
        unicodedata.normalize("NFKD", ent.text.lower())
        for ent in answer_doc.ents
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "NORP"}
    }
    if not answer_ents:
        return 0.0
    grounding_lower = grounding.lower()
    mismatched = sum(1 for ent in answer_ents if ent not in grounding_lower)
    return round(mismatched / len(answer_ents), 4)


def feature_contradiction_score(answer: str, grounding: Optional[str]) -> float:
    """NLI contradiction probability between grounding (premise) and answer (hypothesis)."""
    if not grounding:
        return 0.0
    try:
        pipe = _get_nli_pipeline()
        # Truncate to avoid token limits
        premise = grounding[:512]
        hypothesis = answer[:256]
        result = pipe(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)
        label = result[0]["label"].lower()
        score = result[0]["score"]
        if "contradiction" in label:
            return round(score, 4)
        elif "entailment" in label:
            return round(1.0 - score, 4)
        else:
            return 0.3  # neutral
    except Exception:
        return 0.0


def feature_numeric_mismatch(answer: str, grounding: Optional[str]) -> float:
    """Fraction of numbers in answer that don't appear in grounding text."""
    numbers_in_answer = re.findall(r"\b\d[\d,\.]*\b", answer)
    if not numbers_in_answer or not grounding:
        return 0.0
    mismatched = sum(
        1 for n in numbers_in_answer if n.replace(",", "") not in grounding.replace(",", "")
    )
    return round(mismatched / len(numbers_in_answer), 4)


# ── Main entry point ─────────────────────────────────────────────────────────

def extract_features(question: str, answer: str, grounding: Optional[str] = None) -> dict:
    """
    Returns a dict of all 5 features plus the grounding text used.
    If grounding is None, fetches from Wikipedia.
    """
    if grounding is None:
        grounding = fetch_wikipedia_summary(question)

    return {
        "semantic_sim": feature_semantic_similarity(answer, grounding),
        "confidence_score": feature_confidence_score(answer),
        "ner_mismatch": feature_ner_mismatch(answer, grounding),
        "contradiction_score": feature_contradiction_score(answer, grounding),
        "numeric_mismatch": feature_numeric_mismatch(answer, grounding),
        "_grounding_snippet": (grounding or "")[:300],
    }


def features_to_array(feat_dict: dict) -> np.ndarray:
    """Extract the 5 numeric features as a numpy array for the classifier."""
    keys = ["semantic_sim", "confidence_score", "ner_mismatch", "contradiction_score", "numeric_mismatch"]
    return np.array([feat_dict[k] for k in keys], dtype=np.float32)
