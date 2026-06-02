# Section B - Retrieval pipeline

Baseline implementation for Project A, Section B. The current pipeline uses one
embedding per page and brute-force dot-product retrieval over the submitted
artifacts.

## Current public result

Last local public evaluation:

```text
public_queries=50
mean_ndcg@10=0.1288
query_phase_time=1.24s
```

## Setup

```bash
cd section_b
pip install -r requirements.txt
```

For local index rebuilding, the corpus should live at
`data/Wikipedia Entries/` as provided in the handout.

## Artifacts

The submitted baseline index is stored under `artifacts/`:

- `artifacts/index_vectors.npy`: L2-normalized MiniLM page embeddings,
  shape `(6742, 384)`, dtype `float32`.
- `artifacts/index_meta.json`: `page_id` mapping and chunk metadata.

The baseline uses `sentence-transformers/all-MiniLM-L6-v2`, as required.

## Build index (offline, not timed)

Run once locally to recreate `artifacts/`. Staff do not rebuild the index at
grading time; `run()` loads these files from disk.

```bash
python scripts/build_index.py
```

## Public self-test

After building, verify a fresh run loads your submitted artifacts without
rebuilding the index:

```bash
python scripts/eval_public.py
```

If the model is already cached and the environment blocks internet access, run:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python scripts/eval_public.py
```

## Pipeline files

- `main.py`: autograder entry point, exposes `run(queries)`.
- `chunk.py`: page-to-chunk conversion. The current baseline uses one chunk per
  page.
- `embed.py`: MiniLM embedding helpers.
- `index.py`: offline artifact creation and artifact loading.
- `retrieve.py`: query-time retrieval and ranking.
- `utils.py`: shared paths and JSON loading helpers.
- `eval.py`, `scripts/*.py`: read-only evaluation/build scripts from the
  handout.
