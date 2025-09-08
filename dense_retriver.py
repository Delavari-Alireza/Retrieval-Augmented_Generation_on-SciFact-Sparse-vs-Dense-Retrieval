import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
import faiss


def _compose_text(meta: dict) -> str:
    title = meta.get("title") or ""
    body  = meta.get("text") or ""
    if not body and "abstract" in meta:
        ab = meta["abstract"]
        body = " ".join(ab) if isinstance(ab, list) else str(ab)
    return (title + " " + body).strip()

data_dir   = "datasets/scifact"
out_path   = "results/dense_results.json"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
topk       = 100
batch_size = 128
device     = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    
#Load SciFact (full corpus + test queries/qrels)

# dataset = "scifact"
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# data_path = util.download_and_unzip(url, "datasets")
# corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="test")

#Prepare corpus texts and IDs
doc_ids = list(corpus.keys())
doc_texts = [_compose_text(corpus[did]) for did in doc_ids]

#Load encoder and embed corpus (normalize => cosine via inner product)
model = SentenceTransformer(model_name, device=device)
doc_embs = model.encode(
    doc_texts,
    batch_size=batch_size,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype(np.float32)

#Build FAISS IP index (cosine because we normalized)
dim = doc_embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(doc_embs)

#Embed queries and retrieve top-K
qids = list(queries.keys())
qtexts = [queries[qid] for qid in qids]
q_embs = model.encode(
    qtexts,
    batch_size=batch_size,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype(np.float32)

D, I = index.search(q_embs, topk)  # similarities + indices

#Pack results in BEIR format: {qid: {docid: score, ...}}
results = {}
for row, qid in enumerate(tqdm(qids, desc="Dense retrieve")):
    res = {doc_ids[int(I[row, col])]: float(D[row, col]) for col in range(I.shape[1])}
    results[qid] = res


Path("results").mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)
print(f"Saved {out_path} with {len(results)} queries and top-{topk} docs each.")