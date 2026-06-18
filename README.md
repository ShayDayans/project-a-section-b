# Project B - Wikipedia Retrieval
End-to-end page retrieval system for the course project.

Link to the video: https://technionmail-my.sharepoint.com/:f:/g/personal/shay_dayan_campus_technion_ac_il/IgBOfEl6DJDiToeAKrCB3QjVAZXNJmxwzoyE7jy-D3qe4rI?e=eFto5j 

The autograder imports:
`main.py` and calls:

```python
run(queries: list[str]) -> list[list[int]]
```

For each query, the system returns a ranked list of Wikipedia `page_id` values.
Scoring is page-level NDCG@10.

The retrieval pipeline combines dense retrieval, BM25 retrieval, and a final
cross-encoder reranking stage. Dense embeddings use
`sentence-transformers/all-MiniLM-L6-v2`, and the reranker uses
`cross-encoder/ms-marco-MiniLM-L12-v2`.

## What This Repository Contains

- `main.py`: autograder entry point. `run()` loads committed artifacts and runs retrieval.
- `retrieve.py`: query-time hybrid retrieval, reranking, and page aggregation.
- `index.py`: offline artifact creation and artifact loading.
- `embed.py`: embedding utilities for `sentence-transformers/all-MiniLM-L6-v2`.
- `chunk.py`: chunking logic and chunk-to-page mapping.
- `utils.py`: shared helpers and path definitions.
- `scripts/eval_public.py`: evaluates the system on the 50 public queries.
- `scripts/build_index.py`: offline artifact rebuild script.
- `data/Wikipedia Entries/`: full corpus.
- `artifacts/`: committed retrieval artifacts required for evaluation.

## Setup on the course GPU machine

From the project root:

```bash
cd ~/project-a-section-b
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Required Artifacts

This repository is designed so that grading does not rebuild the index.
`run()` loads committed files from `artifacts/` at query time.

Important committed artifacts currently include:

- `artifacts/dense_chunks.faiss`: FAISS dense index over chunk embeddings.
  Format: a FAISS `IndexFlatIP` index containing the dense chunk vectors used at runtime for nearest-neighbor retrieval.
- `artifacts/dense_chunks_meta.json`: dense chunk metadata and chunk-to-page mapping.
  Format: a JSON object containing model metadata, vector count, and a `chunks` list with fields such as `page_id`, `title`, `chunk_text`, `paragraph_index`, and `subchunk_index`.
- `artifacts/bm25_chunks_arrays.npz`: serialized BM25 posting/statistics arrays.
  Format: a NumPy `.npz` archive containing compact BM25 arrays such as vocabulary terms, posting offsets, document frequencies, IDF values, posting doc indices, term frequencies, and chunk lengths.
- `artifacts/bm25_chunks_meta.json`: BM25 chunk metadata aligned with dense chunks.
  Format: a JSON metadata file containing BM25 configuration, corpus statistics, and the chunk references aligned with the dense chunk order.
- `artifacts/models/all-MiniLM-L6-v2/`: bundled sentence-transformer model files.
  Format: a local `sentence-transformers` checkpoint directory used at runtime to embed queries with `sentence-transformers/all-MiniLM-L6-v2`.
- `artifacts/cross_encoder/ms-marco-MiniLM-L12-v2/`: bundled cross-encoder reranker used at query time.
  Format: a local cross-encoder checkpoint directory used at runtime for the final reranking stage.

If these artifacts are missing, `run()` may fail or the public evaluation may
implicitly depend on internet/model downloads, which is not acceptable for a
fresh grading clone.

## Git LFS

Large artifacts under `artifacts/` should be tracked with Git LFS.

On a fresh machine, after cloning the repository, run:

```bash
git lfs install
git lfs pull
```

This repository is configured to store large retrieval artifacts such as
`.faiss`, `.npz`, `.json`, `.bin`, and `.safetensors` files in `artifacts/`
with Git LFS.

## How To Run The Public Evaluation

From the project root:

```bash
python scripts/eval_public.py
```

This script:

- loads `data/public_queries.json`
- calls `main.run(...)`
- prints `public_queries`, `mean_ndcg@10`, and `query_phase_time`


## Optional Offline Rebuild

Rebuilding the retrieval artifacts is optional and should be treated as an
offline development step, not part of normal evaluation.

```bash
python scripts/build_index.py
```

Use this only if you intentionally want to regenerate the contents of
`artifacts/`.

## Notes

- Required embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Allowed libraries by the assignment: stdlib, `numpy`, `sentence-transformers`, `faiss-cpu`.
- The autograder only calls `run()`.
- Only the top 10 returned page IDs per query affect the score.
