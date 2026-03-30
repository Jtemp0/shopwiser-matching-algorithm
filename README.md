# ShopWiser

Algorithmic grocery product normalisation and clustering for cross-retailer comparison (see [SPECIFICATIONs.md](SPECIFICATIONs.md)).

## Data layout

| Path | Role |
|------|------|
| `data/raw/` | Source CSVs: `raw.csv` (full), `raw_1000.csv` (~1000 rows for dev) |
| `data/processed/` | `normalized_products.csv` (full) or `normalized_products_sample.csv` (`--sample`) |
| `data/outputs/clusters/` | Full run: `clusters.csv`, `cluster_summary.csv`, `singletons.csv`, `audit_sample_50.csv` |
| `data/outputs/clusters_sample/` | Same filenames when you pass `--sample` (does not overwrite full outputs) |
| `data/embeddings/` | Optional embedding caches from `scripts/generate_embeddings.py` |

Paths are resolved from the repository root via [`shopwiser.paths`](src/shopwiser/paths.py) (install the editable package with `uv sync`).

### Full dataset vs sample (`raw_1000`)

Use **`--sample`** on normalisation and clustering so quick runs use `raw_1000.csv` and separate processed/cluster folders. Omit the flag for production runs on `raw.csv`.

| Step | Full (default) | Sample |
|------|----------------|--------|
| Normalise | `uv run python main.py normalise` | `uv run python main.py normalise --sample` |
| Cluster | `uv run python main.py cluster` | `uv run python main.py cluster --sample` |
| Audit | `uv run python main.py audit` | `uv run python main.py audit --sample` |

After a **sample** pipeline, clustering reads `data/processed/normalized_products_sample.csv` and writes under `data/outputs/clusters_sample/`. Run `normalise --sample` before `cluster --sample`.

## Code layout

| Area | Package |
|------|---------|
| Preprocess | [`shopwiser.preprocess`](src/shopwiser/preprocess/) |
| Clustering | [`shopwiser.clustering`](src/shopwiser/clustering/) |
| Similarity tests | [`tests/test_similarity.py`](tests/test_similarity.py) |
| LLM audit | [`shopwiser.audit`](src/shopwiser/audit/) |
| **CLI** | [`main.py`](main.py) at repo root — subcommands `normalise` (alias `norm`), `cluster`, `audit`, `test-similarity` (alias `test`) |

Install once: `uv sync`.

## How to run (end-to-end)

1. **Install:** `uv sync`.
2. **Normalise:** `uv run python main.py normalise` — or `... normalise --sample` for `raw_1000.csv`.
3. **Cluster:** `uv run python main.py cluster` — or `... cluster --sample` to match a sample normalise run.
4. **Optional LLM audit** (`ANTHROPIC_API_KEY` set): `uv run python main.py audit` — add `--sample` if you clustered the sample outputs.
5. **Similarity tests** (no large CSV): `uv run python main.py test-similarity` or `uv run python -m tests.test_similarity` ([`tests/test_similarity.py`](tests/test_similarity.py)).

**Help:** `uv run python main.py --help`, `uv run python main.py cluster --help`.

**Module entrypoints** (same behaviour as before):  
`uv run python -m shopwiser.preprocess.normalise --sample` · `uv run python -m shopwiser.clustering.main --sample`

**Embeddings** (optional): `uv run python scripts/generate_embeddings.py` — defaults to `data/processed/normalized_products.csv`.
