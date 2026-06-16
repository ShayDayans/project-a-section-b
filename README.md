# Project B - Wikipedia Retrieval
End-to-end page retrieval system for the course project.
Link to the video: https://technionmail-my.sharepoint.com/:f:/g/personal/shay_dayan_campus_technion_ac_il/IgBOfEl6DJDiToeAKrCB3QjVAZXNJmxwzoyE7jy-D3qe4rI?e=eFto5j 

The autograder imports
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
cd ~/ProjectB
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Install a PyTorch version that fits the course GPU machine:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip cache purge

python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

Verify that the GPU is available:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

Expected output should include:

```text
cuda build: 12.1
cuda available: True
```

If there is a model version error, update `sentence-transformers`:

```bash
python -m pip install -U "sentence-transformers>=5.5.1"
```

## Required Artifacts

This repository is designed so that grading does not rebuild the index.
`run()` loads committed files from `artifacts/` at query time.

Important committed artifacts currently include:

- `artifacts/dense_chunks.faiss`: FAISS dense index over chunk embeddings.
- `artifacts/dense_chunks_meta.json`: dense chunk metadata and chunk-to-page mapping.
- `artifacts/bm25_chunks_arrays.npz`: serialized BM25 posting/statistics arrays.
- `artifacts/bm25_chunks_meta.json`: BM25 chunk metadata aligned with dense chunks.
- `artifacts/models/all-MiniLM-L6-v2/`: bundled sentence-transformer model files.
- `artifacts/cross_encoder/ms-marco-MiniLM-L12-v2/`: bundled cross-encoder reranker used at query time.

If these artifacts are missing, `run()` may fail or the public evaluation may
implicitly depend on internet/model downloads, which is not acceptable for a
fresh grading clone.

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
