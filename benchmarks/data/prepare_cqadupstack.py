"""
CQADupStack (English) → ColBERT-Zero embeddings.

Uses HuggingFace `datasets` to load BeIR/cqadupstack english subforum.
Duplicate-question detection task — short docs, hard negatives by nature.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "lightonai/ColBERT-Zero"
MAX_DOCS = 5000


def load_cqadupstack() -> Tuple[List[str], List[int], List[str], Dict[int, List[int]]]:
    """Load corpus, queries, qrels from HuggingFace."""
    from datasets import load_dataset

    log.info("Loading BeIR/cqadupstack english...")
    corpus_ds = load_dataset("BeIR/cqadupstack", "english", split="corpus")
    queries_ds = load_dataset("BeIR/cqadupstack", "english", split="queries")
    qrels_ds = load_dataset("mteb/cqadupstack-english", split="test")

    log.info("Corpus: %d docs, Queries: %d, Qrels: %d",
             len(corpus_ds), len(queries_ds), len(qrels_ds))

    qrel_pairs: Dict[str, List[str]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        if row["score"] >= 1:
            qrel_pairs.setdefault(qid, []).append(cid)

    relevant_cids = set()
    for cids in qrel_pairs.values():
        relevant_cids.update(cids)

    corpus_id_to_text: Dict[str, str] = {}
    for row in corpus_ds:
        cid = str(row["_id"])
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        full = f"{title}\n{text}".strip() if title else text
        corpus_id_to_text[cid] = full

    selected_cids = sorted(relevant_cids)
    remaining = MAX_DOCS - len(selected_cids)
    if remaining > 0:
        pool = [cid for cid in corpus_id_to_text if cid not in relevant_cids]
        rng = np.random.RandomState(42)
        extra = rng.choice(pool, min(remaining, len(pool)), replace=False).tolist()
        selected_cids = sorted(set(selected_cids) | set(extra))

    selected_cids = selected_cids[:MAX_DOCS]
    cid_to_idx = {cid: i for i, cid in enumerate(selected_cids)}

    doc_texts = [corpus_id_to_text[cid] for cid in selected_cids]
    doc_ids = list(range(len(doc_texts)))

    query_id_to_text: Dict[str, str] = {}
    for row in queries_ds:
        qid = str(row["_id"])
        query_id_to_text[qid] = row["text"]

    valid_qids = [qid for qid in qrel_pairs if qid in query_id_to_text
                  and any(cid in cid_to_idx for cid in qrel_pairs[qid])]
    valid_qids.sort()

    query_texts = [query_id_to_text[qid] for qid in valid_qids]
    idx_qrels: Dict[int, List[int]] = {}
    for qi, qid in enumerate(valid_qids):
        mapped = [cid_to_idx[cid] for cid in qrel_pairs[qid] if cid in cid_to_idx]
        if mapped:
            idx_qrels[qi] = mapped

    log.info("Selected %d docs, %d queries with %d qrels",
             len(doc_texts), len(query_texts), len(idx_qrels))
    return doc_texts, doc_ids, query_texts, idx_qrels


def embed(doc_texts: List[str], query_texts: List[str]):
    from pylate import models

    log.info("Loading ColBERT-Zero model...")
    model = models.ColBERT(MODEL_NAME)

    log.info("Embedding %d documents...", len(doc_texts))
    doc_embs = model.encode(doc_texts, batch_size=32, is_query=False, show_progress_bar=True)

    log.info("Embedding %d queries...", len(query_texts))
    query_embs = model.encode(query_texts, batch_size=32, is_query=True, show_progress_bar=True)

    doc_vecs = [np.array(e, dtype=np.float16) for e in doc_embs]
    query_vecs = [np.array(e, dtype=np.float16) for e in query_embs]
    return doc_vecs, query_vecs


def save_npz(path: Path, doc_vecs, query_vecs, idx_qrels):
    all_doc = np.vstack([v.astype(np.float16) for v in doc_vecs])
    offsets = []
    pos = 0
    for v in doc_vecs:
        offsets.append((pos, pos + len(v)))
        pos += len(v)

    all_q = np.vstack([v.astype(np.float16) for v in query_vecs])
    q_offsets = []
    pos = 0
    for v in query_vecs:
        q_offsets.append((pos, pos + len(v)))
        pos += len(v)

    max_rels = max((len(v) for v in idx_qrels.values()), default=1)
    qrels_mat = np.full((len(query_vecs), max_rels), -1, dtype=np.int32)
    for qi, rels in idx_qrels.items():
        for ri, idx in enumerate(rels):
            qrels_mat[qi, ri] = idx

    np.savez_compressed(
        path,
        doc_vectors=all_doc,
        doc_offsets=np.array(offsets, dtype=np.int64),
        query_vectors=all_q,
        query_offsets=np.array(q_offsets, dtype=np.int64),
        qrels=qrels_mat,
        doc_ids=np.array([str(i) for i in range(len(doc_vecs))], dtype=object),
        n_docs=np.array(len(doc_vecs)),
        n_queries=np.array(len(query_vecs)),
        dim=np.array(all_doc.shape[1]),
    )
    log.info("Saved %s (%.1f MB)", path, path.stat().st_size / 1024 / 1024)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(
        Path.home() / ".cache/voyager-qa/cqadupstack_english.npz"))
    args = parser.parse_args()

    doc_texts, doc_ids, query_texts, idx_qrels = load_cqadupstack()
    doc_vecs, query_vecs = embed(doc_texts, query_texts)

    avg_vecs = np.mean([len(v) for v in doc_vecs])
    log.info("Doc embeddings: %d docs, avg %.0f vecs/doc, dim %d",
             len(doc_vecs), avg_vecs, doc_vecs[0].shape[1])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, doc_vecs, query_vecs, idx_qrels)
    log.info("Done.")


if __name__ == "__main__":
    main()
