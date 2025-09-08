from beir import util
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
import re
import json, heapq
from tqdm import tqdm


# dataset = "scifact"
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# data_path = util.download_and_unzip(url, "datasets")

data_dir   = "datasets/scifact"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="test")



doc_ids = list(corpus.keys())
doc_texts = []
for did in doc_ids:
    meta = corpus[did]
    title = meta.get("title") or ""
    body  = meta.get("text") or ""
    # (Optional) handle SciFact's 'abstract' if present in this loader:
    if not body and "abstract" in meta:
        ab = meta["abstract"]
        body = " ".join(ab) if isinstance(ab, list) else str(ab)
    doc_texts.append((title + " " + body).strip())




TOKEN = re.compile(r"\w+")
def tok(text: str): return TOKEN.findall((text or "").lower())


tokenized_corpus = [tok(t) for t in doc_texts]
bm25 = BM25Okapi(tokenized_corpus)



topk = 100
# Retrieve for each test query
qids = list(queries.keys())        # test queries
results = {}
for qid in tqdm(qids, desc="BM25 retrieve"):
    qtext = queries[qid]
    qtok = tok(qtext)
    scores = bm25.get_scores(qtok)  # numpy array, len == len(doc_ids)

    # Take top-k indices by score
    idxs = heapq.nlargest(topk, range(len(scores)), key=lambda i: scores[i])
    results[qid] = {doc_ids[i]: float(scores[i]) for i in idxs}  # cast to plain floats


from pathlib import Path
Path("results").mkdir(parents=True, exist_ok=True)
with open("results/sparse_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)
print(f"Saved results/sparse_results.json with {len(results)} queries and top-{topk} docs each.")