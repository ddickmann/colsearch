"""
Fast dataset prep: MS MARCO v2.1 → ColBERT-Zero embeddings.

Uses HuggingFace `datasets` streaming (no 8GB download). Concatenates
the 10 passages per query into ~1024-token documents with built-in
relevance labels.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "lightonai/ColBERT-Zero"
TARGET_N_DOCS = 2500
TARGET_N_QUERIES = 200
MIN_TOKENS = 768
MAX_TOKENS = 1280


def collect_from_hf(tokenizer) -> Tuple[List[str], List[str], Dict[int, List[int]]]:
    """Stream MS MARCO v2.1, concatenate passages into long docs."""
    from datasets import load_dataset

    log.info("Streaming MS MARCO v2.1 from HuggingFace...")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)

    docs: List[str] = []
    queries: List[str] = []
    qrels: Dict[int, List[int]] = {}

    seen_docs: Dict[str, int] = {}
    count = 0

    for row in ds:
        passages = row["passages"]
        texts = passages["passage_text"]
        is_selected = passages["is_selected"]
        query_text = row["query"]

        doc_text = "\n\n".join(texts)
        n_tok = len(tokenizer.tokenize(doc_text))

        if n_tok < MIN_TOKENS or n_tok > MAX_TOKENS:
            count += 1
            if count % 50000 == 0:
                log.info("  scanned %dk rows, %d docs, %d queries",
                         count // 1000, len(docs), len(queries))
            continue

        doc_hash = doc_text[:200]
        if doc_hash in seen_docs:
            doc_idx = seen_docs[doc_hash]
        else:
            doc_idx = len(docs)
            docs.append(doc_text)
            seen_docs[doc_hash] = doc_idx

        if len(queries) < TARGET_N_QUERIES:
            qi = len(queries)
            queries.append(query_text)
            if any(s == 1 for s in is_selected):
                qrels[qi] = [doc_idx]

        count += 1
        if count % 50000 == 0:
            log.info("  scanned %dk rows, %d docs, %d queries",
                     count // 1000, len(docs), len(queries))

        if len(docs) >= TARGET_N_DOCS and len(queries) >= TARGET_N_QUERIES:
            break

    log.info("Collected %d docs (%d-%d tokens), %d queries, %d with qrels",
             len(docs), MIN_TOKENS, MAX_TOKENS, len(queries), len(qrels))
    return docs, queries, qrels


def embed(doc_texts: List[str], query_texts: List[str]):
    """Embed with ColBERT-Zero via pylate."""
    from pylate import models

    log.info("Loading ColBERT-Zero model...")
    model = models.ColBERT(MODEL_NAME)

    log.info("Embedding %d documents...", len(doc_texts))
    doc_embs = model.encode(doc_texts, batch_size=4, is_query=False, show_progress_bar=True)

    log.info("Embedding %d queries...", len(query_texts))
    query_embs = model.encode(query_texts, batch_size=32, is_query=True, show_progress_bar=True)

    doc_vecs = [np.array(e, dtype=np.float16) for e in doc_embs]
    query_vecs = [np.array(e, dtype=np.float16) for e in query_embs]
    return doc_vecs, query_vecs


def save_npz(path: Path, doc_vecs, query_vecs, idx_qrels):
    """Save as .npz matching msmarco_loader.py's _load_from_npz format."""
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

    doc_ids = [str(i) for i in range(len(doc_vecs))]

    np.savez_compressed(
        path,
        doc_vectors=all_doc,
        doc_offsets=np.array(offsets, dtype=np.int64),
        query_vectors=all_q,
        query_offsets=np.array(q_offsets, dtype=np.int64),
        qrels=qrels_mat,
        doc_ids=np.array(doc_ids, dtype=object),
        n_docs=np.array(len(doc_vecs)),
        n_queries=np.array(len(query_vecs)),
        dim=np.array(all_doc.shape[1]),
    )
    log.info("Saved %s (%.1f MB)", path, path.stat().st_size / 1024 / 1024)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path.home() / ".cache/voyager-qa/msmarco_doc_2k5.npz"))
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    doc_texts, query_texts, idx_qrels = collect_from_hf(tokenizer)
    doc_vecs, query_vecs = embed(doc_texts, query_texts)

    avg_tokens = np.mean([len(v) for v in doc_vecs])
    log.info("Doc embeddings: %d docs, avg %.0f tokens/doc, dim %d",
             len(doc_vecs), avg_tokens, doc_vecs[0].shape[1])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, doc_vecs, query_vecs, idx_qrels)
    log.info("Done.")


if __name__ == "__main__":
    main()
