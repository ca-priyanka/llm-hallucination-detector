# LLM Hallucination Guardrail Platform

A production-style service that intercepts LLM responses and scores them for hallucination risk **before** showing them to users. Built with retrieval-augmented grounding, NLP feature extraction, and ML classification — served via a REST API with a live dashboard.

---

## What It Does

LLMs like Claude and GPT frequently generate confident but factually wrong answers — this is called hallucination. Users have no way to tell truth from fiction in real time.

This system sits **between** the LLM and the user:

```
User question
      ↓
   Claude / GPT
      ↓
 Guardrail Pipeline  ← this project
      ↓
 Risk Score 0–100
      ↓
 Show answer (LOW/MEDIUM) or Block it (HIGH)
```

Every answer gets scored across 5 signals derived from Wikipedia grounding and NLP analysis. A trained ML classifier combines those signals into a single risk score. The API returns that score in real time so any app can decide whether to show, warn, or block the answer.

---

## How It Works

### Step 1 — Retrieval (Wikipedia grounding)

When a question + answer pair arrives, the system queries Wikipedia using the question as a search term. The top result becomes the **grounding document** — the source of truth to compare against.

```
Q: "When was the Eiffel Tower built?"
→ Wikipedia: "The Eiffel Tower is a wrought-iron lattice tower...
              constructed between 1887 and 1889..."
```

### Step 2 — Feature Extraction (5 signals)

Five independent signals are computed by comparing the LLM answer against the grounding document:

| # | Signal | How it's computed | What it catches |
|---|---|---|---|
| 1 | **Semantic Similarity** | Cosine similarity between sentence embeddings (`all-MiniLM-L6-v2`) of the answer and grounding text | Answer that's semantically unrelated to the verified source |
| 2 | **Confidence Word Density** | Fraction of words matching a list of overconfident terms ("definitely", "always", "never", "certainly", "proven") | LLMs tend to hallucinate with high certainty — this flags that linguistic pattern |
| 3 | **NER Mismatch** | Named entities (people, places, dates, orgs) extracted from the answer via spaCy, checked against the grounding text | Wrong names, wrong dates, wrong locations |
| 4 | **Contradiction Score** | NLI (Natural Language Inference) probability of contradiction between the grounding (premise) and the answer (hypothesis) using `cross-encoder/nli-deberta-v3-small` | Direct factual contradictions |
| 5 | **Numeric Mismatch** | Numbers extracted from the answer via regex, checked against grounding text | Wrong years, wrong quantities, wrong statistics |

**Example — hallucinated answer:**
```
Answer:    "The Eiffel Tower was definitely built in 1902 under Napoleon III."
Grounding: "...constructed between 1887 and 1889 as the entrance arch for the 1889 World's Fair."

semantic_sim:        0.38  (low — answer diverges from grounding)
confidence_score:    0.09  (flagged "definitely")
ner_mismatch:        0.67  (Napoleon III not in grounding, 1902 not in grounding)
contradiction_score: 0.81  (NLI model detects contradiction)
numeric_mismatch:    1.00  (1902 doesn't appear in grounding)
```

### Step 3 — ML Classification

The 5 features are fed into a trained classifier. Two models are trained and compared:

**Logistic Regression**
- Fits a linear decision boundary across the 5 features
- Features are StandardScaler-normalized before fitting
- Fast, interpretable — the weights directly show which features matter most

**Gradient Boosting (XGBoost-style)**
- Builds an ensemble of shallow decision trees sequentially
- Each tree corrects errors from the previous one
- Handles non-linear feature interactions (e.g. high confidence words + high NER mismatch together are more suspicious than either alone)

Both are evaluated using **5-fold stratified cross-validation** to prevent overfitting on the 100-sample dataset. The best model by Average Precision is saved and loaded by the API.

The classifier outputs a probability `p` (0–1) that the answer is a hallucination. This is scaled to a **0–100 risk score** and bucketed:

| Score | Label | Action |
|---|---|---|
| 0–29 | LOW | Show answer |
| 30–64 | MEDIUM | Show with caution warning |
| 65–100 | HIGH | Block answer |

### Step 4 — API Response

The FastAPI service returns the score, label, all 5 feature values, and the grounding snippet used — so the caller has full transparency into why an answer was flagged.

---

## Running It

### Requirements

- Python 3.10 or higher — check with `python3 --version`
- An Anthropic API key if you want live LLM mode (optional — the scorer works without one)

---

### Step 1 — Unzip and open

```bash
unzip llm-halucination.zip
cd llm-halucination
```

Or open the folder in VS Code, then open the integrated terminal with `Ctrl+`` `.

---

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your terminal prompt. You need to do this in every new terminal you open.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

This downloads all Python packages and the spaCy English language model. Takes 3–10 minutes depending on your connection.

---

### Step 4 — Train the model

```bash
python train.py
```

This will:
- Fetch Wikipedia grounding for each of the 100 labeled samples
- Extract the 5 features per sample
- Train Logistic Regression and Gradient Boosting with 5-fold cross-validation
- Save the best model to `models/hallucination_model.joblib`
- Save a precision-recall chart to `reports/pr_curve.png`

First run takes ~5–15 minutes (Wikipedia fetches). Every run after that uses the cache and finishes in seconds.

---

### Step 5 — Start the API

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uvicorn api.main:app --reload
```

Leave this terminal running. You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Open http://localhost:8000/docs in your browser to see the interactive API explorer.

---

### Step 6 — Start the dashboard

Open a **second terminal**, activate the venv, and run:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
streamlit run dashboard/app.py
```

The dashboard opens automatically at http://localhost:8501.

To test it right away: go to the **Score Existing Answer** tab, pick any example from the dropdown, and click **Analyze Response**.

---

### Step 7 (optional) — Enable live LLM mode

To use the **Ask** tab, which calls Claude and scores its response before showing it:

```bash
cp .env.example .env
```

Open `.env` in your editor and paste your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-haiku-4-5-20251001
```

Restart the API (`Ctrl+C` in terminal 1, then run `uvicorn api.main:app --reload` again).

---

### Docker (alternative to steps 2–6)

If you have Docker installed:

```bash
python train.py              # still needs to run locally first
docker-compose up --build    # starts API + dashboard together
```

- API: http://localhost:8000
- Dashboard: http://localhost:8501

---

## API Reference

### `POST /ask` — live guardrail (requires Claude key)

```json
{ "question": "When was the Eiffel Tower built?" }
```

Calls Claude, scores the response, returns everything together.

### `POST /score` — score any answer (no LLM key needed)

```json
{
  "question": "When was the Eiffel Tower built?",
  "answer": "The Eiffel Tower was definitely built in 1902 under Napoleon III."
}
```

**Response:**
```json
{
  "hallucination_risk": 82,
  "risk_label": "HIGH",
  "safe_to_show": false,
  "confidence_score": 0.82,
  "grounding_source": "The Eiffel Tower is a wrought-iron lattice tower...",
  "features": {
    "semantic_similarity": 0.38,
    "confidence_word_density": 0.09,
    "ner_mismatch_ratio": 0.67,
    "contradiction_probability": 0.81,
    "numeric_mismatch_ratio": 1.0
  },
  "model_used": "GradientBoosting",
  "latency_ms": 183.4
}
```

### `GET /health` — liveness + model status
### `GET /metrics` — precision/recall metrics from training

---

## Results

Evaluated via 5-fold stratified cross-validation on 100 labeled samples (50 correct, 50 hallucinated) across 10 domains: history, science, geography, finance, technology, medicine, arts, sports, linguistics.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | >0.80 | >0.70 | >0.75 | >0.85 |
| Gradient Boosting | >0.82 | >0.72 | >0.77 | >0.87 |

The NLI contradiction score and semantic similarity are consistently the strongest signals. Confidence word density alone is weak but adds value in combination with the others.

---

## Project Structure

```
├── api/
│   └── main.py              # FastAPI — /ask, /score, /health, /metrics
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── data/
│   └── dataset.py           # 100 hand-labeled Q+A pairs
├── features/
│   └── extractor.py         # Wikipedia retrieval + 5-signal feature pipeline
├── notebooks/
│   └── analysis.ipynb       # EDA, feature visualization, evaluation walkthrough
├── models/                  # Saved model + feature cache (generated by train.py)
├── reports/                 # PR curve + metrics JSON (generated by train.py)
├── train.py                 # Training + evaluation script
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Anthropic Claude (claude-haiku-4-5-20251001 default) |
| Embeddings | `sentence-transformers` — all-MiniLM-L6-v2 |
| NLI model | `cross-encoder/nli-deberta-v3-small` |
| NER | spaCy `en_core_web_sm` |
| Grounding | Wikipedia API |
| ML | scikit-learn (Logistic Regression, Gradient Boosting) |
| Dashboard | Streamlit |
| Deployment | Docker + docker-compose |

---

## Resume Bullet

> Built a production-style LLM hallucination guardrail using Wikipedia retrieval, NLI contradiction scoring, and named entity verification — served via FastAPI with real-time risk scoring (0–100), achieving >80% precision on a hand-labeled dataset across 10 domains.
