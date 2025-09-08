# Retrieval‑Augmented Generation on **SciFact**: Sparse vs Dense Retrieval

**Author:** Alireza Delavari
**Focus:** Practical retrieval comparison for RAG grounding (BM25 vs Sentence‑Transformers + FAISS) on BEIR/SciFact.

---

## Abstract

We implement two canonical retrievers—**BM25** (sparse) and **MiniLM + FAISS** (dense)—on the **SciFact** dataset and evaluate them with standard IR metrics (nDCG\@10, Recall\@100, MAP, P\@k). Our goal is not only to compare retrieval quality, but to frame the results in terms of **Retrieval‑Augmented Generation (RAG)** design. Dense retrieval improves **evidence coverage** (Recall\@100), while BM25 retains a slight advantage in **early ranking quality** on technical, term‑heavy queries. We summarize practical design choices for plugging these retrievers into a RAG pipeline and outline a hybrid path that combines their strengths.

---

## Dataset & Setup

* **Benchmark:** BEIR **SciFact** (fact‑checking; \~5k abstracts).
* **Split:** Test queries with full corpus available for retrieval.
* **Outputs:** For each query, top‑100 document IDs + scores in BEIR‑compatible JSON.
* **Evaluation:** Provided `evaluation.py` (nDCG\@10, Recall\@100, MAP, P\@k).

---

## Methods

### Sparse (BM25)

* **Indexing:** `rank-bm25` over `title + text` (fallback to `abstract` when needed).
* **Scoring:** Probabilistic term weighting with TF saturation and length normalization.
* **Strengths:** Fast, interpretable, robust on domain‑specific terminology.

### Dense (Sentence‑Transformers + FAISS)

* **Encoder:** `all‑MiniLM‑L6‑v2` (384‑dim) with **L2‑normalized** embeddings.
* **Index:** FAISS **Inner Product** (cosine similarity via normalization).
* **Strengths:** Captures semantic matches and paraphrases beyond exact token overlap.

---


# Project Structure

```
.
├─ datasets/                 # created by download_data.py (SciFact via BEIR)
├─ results/                  # retrieval outputs (JSON)
│  ├─ sparse_results.json
│  └─ dense_results.json
├─ download_data.py          # downloads & unzips SciFact into datasets/scifact
├─ sparse_retriver.py        # BM25 retriever → results/sparse_results.json
├─ dense_retriver.py         # MiniLM + FAISS retriever → results/dense_results.json
├─ evaluation.py             # standardized metrics (nDCG, Recall, MAP, P@k)
├─ report.pdf                # RAG-focused mini-report (included for review)
├─ README.md
├─ requirements.txt
```

---

# Setup & Run — Step‑by‑Step

## 1) Environment

```bash
python -m venv .venv
source ./.venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Download the dataset (SciFact)

```bash
python download_data.py
# → datasets/scifact
```

## 3) Generate retrieval results

**Sparse (BM25)**

```bash
python sparse_retriver.py
# writes: results/sparse_results.json
```

**Dense (MiniLM + FAISS)**

```bash
python dense_retriver.py
# writes: results/dense_results.json
```

## 4) Evaluate with the provided script

```bash
# Evaluate the sparse retriever
python evaluation.py datasets/scifact results/sparse_results.json

# Evaluate the dense retriever
python evaluation.py datasets/scifact results/dense_results.json
```

---


## Results (SciFact, test split)

| Retriever                  |   nDCG\@10 |  nDCG\@100 | MAP\@10 |   MAP\@100 | Recall\@10 | Recall\@100 |      P\@10 |     P\@100 |
| -------------------------- | ---------: | ---------: | ------: | ---------: | ---------: | ----------: | ---------: | ---------: |
| **BM25 (sparse)**          | **0.6519** |     0.6759 |  0.6070 | **0.6132** |     0.7740 |      0.8731 |     0.0850 |     0.0098 |
| **MiniLM + FAISS (dense)** |     0.6451 | **0.6767** |  0.5959 |     0.6031 | **0.7833** |  **0.9250** | **0.0883** | **0.0105** |

**Takeaways**

* **Dense** substantially improves **Recall\@100** (+0.052), i.e., **more relevant evidence** enters the candidate pool—critical for RAG.
* **BM25** slightly leads on **early ranking** (nDCG/MAP\@10), reflecting SciFact’s technical lexical overlap.
* **Precision\@10** is marginally higher for dense, indicating a few more relevant items at the very top.

---

# Brief Discussion of Results

**SciFact (test split)**

* **BM25 (sparse):** nDCG\@10 **0.6519**, Recall\@100 **0.8731**, MAP\@100 **0.6132**, P\@10 **0.0850**
* **MiniLM + FAISS (dense):** nDCG\@10 **0.6451**, Recall\@100 **0.9250**, MAP\@100 **0.6031**, P\@10 **0.0883**

**Which performed better? Why?**

* **Dense** retrieval achieved **higher Recall\@100** (+0.052), meaning it surfaced **more relevant evidence** overall—valuable for RAG grounding and coverage.
* **BM25** was slightly stronger on **early ranking quality** (nDCG/MAP\@10), likely due to SciFact’s term‑dense scientific queries where exact lexical matching is highly informative.

**Performance trade‑offs**

* **Speed:** BM25 indexing and query scoring are CPU‑fast. Dense has a one‑time **embedding** cost; FAISS search is then very fast.
* **Memory:** At SciFact scale, dense embeddings are modest (\~**8 MB**: 5,183 × 384 × 4 bytes). BM25 index is also lightweight.
* **Retrieval quality:** Dense boosts **semantic recall** (paraphrases/synonyms). BM25 offers excellent **precision at the top** for technical keyword overlap. A simple **hybrid** (combining scores) often raises nDCG without losing recall.

> Both retrievers are provided with identical output formats, enabling plug‑and‑play evaluation and easy integration into a downstream RAG pipeline.


## Implementation Notes (concise)

* **Sparse:** `sparse_retriver.py` → `results/sparse_results.json` (BEIR format).
* **Dense:** `dense_retriver.py` → `results/dense_results.json` (BEIR format).
* **Eval:** `python evaluation.py datasets/scifact <results.json>` yields comparable metrics across systems.

---

## Limitations & Next Steps

* Add a **cross‑encoder re‑ranker** on top‑200 to optimize early precision for RAG.

* Integrate the retriever into an **end‑to‑end RAG** demo (RAG‑Token/Sequence) with citation‑style outputs.

---

## References (selected)

* Robertson & Zaragoza (2009) — BM25 and the probabilistic relevance framework.
* Karpukhin et al. (2020) — Dense Passage Retrieval (bi‑encoder + FAISS).
* Lewis et al. (2020) — Retrieval‑Augmented Generation (RAG) for knowledge‑intensive NLP.
