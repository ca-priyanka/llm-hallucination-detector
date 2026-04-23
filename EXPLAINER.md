# LLM Hallucination Guardrail — Project Explainer

*Written for ML students who want to understand and replicate this project.*

---

## The Problem

Large Language Models (LLMs) like ChatGPT and Claude are trained to sound confident and fluent. The problem is they generate text based on statistical patterns — not a live lookup of facts. So when they don't know something, they don't say "I don't know." They make something up, in the same confident tone as when they're correct.

This is called **hallucination**.

Example:
> **Q:** When was the Eiffel Tower built?
> **A (hallucinated):** "The Eiffel Tower was definitely constructed in 1902 under Napoleon III."

The answer is wrong (it was 1889), but it reads as completely authoritative. A user has no way to tell.

This project builds a system that **automatically detects** when an LLM answer is likely hallucinated, before the user ever sees it.

---

## The Core Idea

If an LLM answer contradicts what a trusted source says about the same topic — it's probably wrong.

So for every answer, we:
1. Look up the topic on Wikipedia (trusted source)
2. Compare the answer against what Wikipedia says
3. Measure several signals of disagreement
4. Feed those signals into a trained classifier
5. Output a risk score: 0 (definitely correct) to 100 (likely hallucination)

---

## The 5 Signals (Features)

These are the inputs to the ML model. Each captures a different type of evidence that an answer might be wrong.

### 1. Semantic Similarity
**What it is:** How similar is the meaning of the LLM answer to the Wikipedia text?

**How it works:** Both the answer and the Wikipedia summary are converted into vectors (lists of numbers) using a model called `sentence-transformers/all-MiniLM-L6-v2`. This model was trained to put sentences with similar meanings close together in vector space. We then compute **cosine similarity** — a number between 0 and 1 where 1 means identical meaning and 0 means completely unrelated.

**Why it matters:** A correct answer about the Eiffel Tower should be semantically close to Wikipedia's description of the Eiffel Tower. A hallucinated answer will diverge.

**Libraries:** `sentence-transformers`

---

### 2. Confidence Word Density
**What it is:** How many overconfident words does the answer contain?

**How it works:** We check for words like "definitely", "always", "never", "certainly", "absolutely", "proven", "obviously". We count how many appear relative to the total word count.

**Why it matters:** Hallucinations often come with false certainty. Correct answers tend to be more measured ("approximately", "around", "generally"). This is a purely linguistic signal — no external knowledge needed.

**Libraries:** Plain Python string matching

---

### 3. Named Entity Mismatch (NER)
**What it is:** Are the specific people, places, dates, and organisations in the answer actually mentioned in the trusted source?

**How it works:** We use **spaCy**, an NLP library, to extract Named Entities from the LLM answer — things like "Napoleon III" (PERSON), "1902" (DATE), "Paris" (GPE). We then check how many of those entities appear in the Wikipedia grounding text. If an entity is in the answer but not in the source, it's suspicious.

**Why it matters:** This directly catches the most common type of hallucination — wrong names, wrong dates, wrong places. "Napoleon III built the Eiffel Tower" will flag because Napoleon III doesn't appear in the Wikipedia article about the Eiffel Tower's construction.

**Libraries:** `spacy` with the `en_core_web_sm` model

---

### 4. Contradiction Score (NLI)
**What it is:** Does the answer logically contradict the Wikipedia source?

**How it works:** This uses **Natural Language Inference (NLI)** — a technique where a model is given two pieces of text and asked: does the second *entail*, *contradict*, or is it *neutral* to the first?

We use `cross-encoder/nli-deberta-v3-small`, a model fine-tuned for this task. We pass:
- **Premise:** the Wikipedia summary
- **Hypothesis:** the LLM answer

If it predicts "contradiction" with high confidence, that's a strong hallucination signal.

**Why it matters:** This is the most powerful signal. Unlike the others which look for surface-level mismatches, NLI understands meaning. "Built in 1902" directly contradicts "built in 1889" — and the model catches that.

**Libraries:** `transformers` (HuggingFace)

---

### 5. Numeric Mismatch
**What it is:** Are the specific numbers in the answer present in the trusted source?

**How it works:** We extract all numbers from the answer using a regular expression (`\b\d[\d,\.]*\b`). We then check each one against the Wikipedia text. Numbers that appear in the answer but not the source are flagged.

**Why it matters:** Hallucinated answers often get numbers slightly wrong — wrong year, wrong quantity, wrong percentage. This is a cheap, fast check that catches many common cases.

**Libraries:** Python `re` (regex)

---

## The ML Model

Once we have the 5 feature values for a given answer, we need to combine them into a single score. That's the job of the classifier.

We train two models and pick whichever performs better:

### Logistic Regression
Think of this as learning a weighted formula:

```
risk = w1 * semantic_sim + w2 * confidence_density + w3 * ner_mismatch + ...
```

During training it finds the weights `w1, w2...` that best separate hallucinations from correct answers. It's fast, interpretable, and works well when features are roughly linear.

### Gradient Boosting
This builds a sequence of small decision trees. Each tree looks at where the previous trees made mistakes and tries to correct them. The final prediction is the combined vote of all trees.

It handles non-linear patterns — for example: "confidence words alone are weak evidence, but confidence words AND numeric mismatch together are very suspicious." Logistic Regression can't learn that interaction; Gradient Boosting can.

### Training & Evaluation
- **Dataset:** 100 hand-labeled question+answer pairs across 10 domains (history, science, geography, finance, technology, medicine, arts, sports, linguistics) — 50 correct, 50 hallucinated
- **Evaluation:** 5-fold stratified cross-validation (splits data 5 ways, trains on 4 folds, tests on 1, rotates — gives a reliable estimate without wasting data on a held-out test set)
- **Metrics:** Precision and Recall on the hallucination class. We target Precision > 80% (when we flag something, we're usually right) and Recall > 70% (we catch most hallucinations).

---

## The System Architecture

```
           ┌──────────────┐
User ───▶  │  Your App    │
           └──────┬───────┘
                  │ question
                  ▼
           ┌──────────────┐
           │  Claude API  │  (or any LLM)
           └──────┬───────┘
                  │ answer
                  ▼
    ┌─────────────────────────────┐
    │      Guardrail Pipeline     │
    │                             │
    │  1. Wikipedia lookup        │
    │  2. Extract 5 features      │
    │  3. ML classifier           │
    │  4. Risk score 0–100        │
    └─────────────┬───────────────┘
                  │
          ┌───────┴────────┐
          │                │
        LOW/MEDIUM        HIGH
          │                │
     Show answer       Block answer
```

---

## The API

Built with **FastAPI** — a Python web framework. It exposes two main endpoints:

- **`POST /ask`** — you send a question, it calls Claude, scores the response, returns everything
- **`POST /score`** — you send a question + answer you already have, it just scores it

The response includes the risk score, the label (LOW/MEDIUM/HIGH), all 5 feature values, and the Wikipedia snippet used — so you can always explain why an answer was flagged.

---

## The Dashboard

Built with **Streamlit** — a Python library that turns scripts into web apps with almost no frontend code. It has two tabs:

- **Ask tab** — live: type a question, Claude answers, guardrail scores it, shows or blocks it
- **Score tab** — manual: paste any question + answer and score it

---

## How to Replicate This Project

If you want to build this yourself, here's the learning path:

| Step | What to learn | Resource to start |
|---|---|---|
| 1 | Python basics | Any Python tutorial |
| 2 | pandas + numpy | Kaggle micro-courses |
| 3 | scikit-learn (Logistic Regression, train/test split, cross-validation) | scikit-learn user guide |
| 4 | Sentence embeddings | `sentence-transformers` docs |
| 5 | Named Entity Recognition | spaCy "101" tutorial |
| 6 | NLI / transformers | HuggingFace "NLI" tutorial |
| 7 | FastAPI basics | FastAPI official tutorial (1 hour) |
| 8 | Streamlit basics | Streamlit "get started" docs |

You don't need to know all of this upfront. Start with steps 1–3 to get the ML classifier working, then layer in the NLP pieces one at a time.

---

## Key Concepts Glossary

| Term | Plain English |
|---|---|
| **Hallucination** | When an LLM confidently states something false |
| **Grounding** | Anchoring an answer to a real, trusted source |
| **Embedding** | Converting text into a list of numbers that captures its meaning |
| **Cosine Similarity** | A measure of how close two vectors are in direction (0 = unrelated, 1 = same meaning) |
| **NER (Named Entity Recognition)** | Automatically labelling words as names, places, dates, organisations, etc. |
| **NLI (Natural Language Inference)** | Determining if one sentence logically entails, contradicts, or is neutral to another |
| **Cross-validation** | A technique to reliably evaluate a model when you have limited data |
| **Precision** | Of everything we flagged as hallucination — how often were we right? |
| **Recall** | Of all the actual hallucinations — how many did we catch? |
| **REST API** | A way to expose functionality over the web via HTTP requests |
