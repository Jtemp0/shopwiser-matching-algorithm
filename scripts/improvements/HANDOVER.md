# ShopWiser Matching Pipeline — v13 Hand-back

End-to-end regeneration of the cluster deliverable in response to the
external review, with three rounds of revisions:

1. **v11 (Plan B re-run)** — re-ran the full pipeline with vocab fixes,
   no LLM stages.
2. **v12 (cleanup + refactor)** — single-source-of-truth conflict
   vocabulary in `shopwiser/conflict_tokens.py`, plurals normalised,
   wine/pâté/confectionery tokens added, codebase cleanup.
3. **v13 (Darius-spec alignment)** — folder rename
   (`clustering`→`rule_matcher`, `ml_matching`→`ml_matcher`), enforced
   the previously dead `MARGIN_THRESHOLD` constant per Darius's spec,
   wrote a point-by-point comparison vs his architecture proposal.

The **canonical shipped CSV** after the strict ensemble + repo cleanup is  
`data/outputs/ensemble/ensemble_clusters.csv` (written by `main.py ensemble`).

The **v13 milestone** table below is kept as a historical snapshot (margin gate,
pre–strict Kruskal gates); counts differ from the current file.

---

## Headline numbers

Comparison against the original deliverable from the contractor's report
(the one Darius reviewed) and the intermediate runs:

| Metric | Original v9 (with LLM) | v11 (no LLM) | v12 (cleanup) | **v13 (margin gate)** |
|---|---:|---:|---:|---:|
| Total clusters | 12,449 | 13,172 | 13,146 | **13,117** |
| 4-way clusters | 6,073 | 4,778 | 4,240 | **4,226** |
| 3-way clusters | 3,683 | 3,999 | 4,145 | **4,145** |
| 2-way clusters | 2,693 | 4,395 | 4,761 | **4,746** |
| Products covered | 40,727 (62.6%) | 39,899 (61.4%) | 38,917 (59.9%) | **38,831 (59.7%)** |
| Probe pass rate, 4-way | 29.7% | 37.9% | 41.6% | **41.6%** |
| Probe pass rate, overall | 45.3% | 52.7% | 56.0% | **55.9%** |
| Calibrated precision, 4-way | 40.8% | 46.5% | 49.1% | **49.1%** |
| Calibrated precision, overall | 51.7% | 56.9% | 59.2% | **59.1%** |
| Audit clean-cluster rate | n/a | n/a | 92.1% | **92.1%** |
| Hard-conflict clusters | 292 | 111 | 3 | **3** |

The 4-way drop vs the original is expected — the original v9 was
inflated by an LLM completion stage that promoted 2/3-way clusters into
4-way. We left the LLM stage off so the deliverable is fully
deterministic. Every quality metric on the same probe Darius used is
meaningfully higher than v9.

v13 vs v12 deltas are tiny (margin gate is a precision-conservatism
boost; most ambiguous edges were already being caught by other gates).
The reason to ship v13 is the principled completeness: every part of
Darius's pipeline spec is now implemented.

## Direct fixes for Darius's specific examples

All five clusters Darius singled out as failures in his REVIEW.md §4 are
**fixed** in v13:

| REVIEW.md cluster | What was wrong | v13 status |
|---|---|---|
| #41 Starbucks Colombia + Tesco Sunset Hour | unrelated coffees grouped | ✓ fixed |
| #6933 Vegetable Spring Rolls + Trex Vegetable Fat | "Vegetable" surface match | ✓ fixed |
| #63 Jimmy's Iced Coffee + Pina Colada | both retailers labelled "Original" | ✓ fixed |
| #4023 Spreadable Goats Cheese + Ardennes Pâté | "Spreadable" surface match | ✓ fixed (`pate` token) |
| #5445 Pringles Sour Cream & Onion + Pringles Cheese & Onion | flavour mismatch | ✓ fixed (`sour` + `cheddar` tokens) |

Verified by `scripts/audit_clusters.py` and direct cluster inspection.

## What changed in v13

### Folder rename for parallel naming
- `src/shopwiser/clustering/` → `src/shopwiser/rule_matcher/`
- `src/shopwiser/ml_matching/` → `src/shopwiser/ml_matcher/`

Both names now describe two matchers built around the same
retrieve→gate→score loop, differing only in which signal dominates
(rule-based vs supervised). All imports updated; tests pass.

### `MARGIN_THRESHOLD` is now actually enforced
The constant `MARGIN_THRESHOLD = 0.04` was defined in
`ml_matcher/config.py` but never read by any code. v13 makes it real:
after argmax per `(anchor, target_supermarket)`, we require the top-1
match score to lead the runner-up by ≥ 0.04 — Darius's "decisive
winner" criterion (§2.6 of his spec).

Effect on the v13 run:
- Margin gate rejected **22,090 / 82,426 (27%)** ambiguous above-threshold edges
- Coverage: 38,831 products (–86 vs v12)
- 4-way: 4,226 (–14 vs v12)

A point-by-point comparison of our pipeline against Darius's
`ShopWiser_Pipeline.pdf` is in
[`COMPARISON_WITH_DARIUS_SPEC.md`](./COMPARISON_WITH_DARIUS_SPEC.md).

## What was refactored (v12)

### `src/shopwiser/conflict_tokens.py` (single source of truth)

Both pipelines now import from one place. Contains:
- `HARD_CONFLICT_NORM` — synonym + plural normalisation
  (`raspberries → raspberry`, `tomatoes → tomato`, `decaffeinated → decaf`, …)
- `FLAVOR_NAMED_TOKENS` — fruit / herb / meat / sauce / wine grape varieties
- `ONE_SIDED_CONFLICT_TOKENS` — tokens whose asymmetric presence is a clash
- `MILK_BASE_TOKENS`, `COOKING_STATE_TOKENS`, `PACKAGING_FORMAT_TOKENS`,
  `PREPARATION_CONFLICT_PAIRS` — mutually-exclusive groups
- `check_hard_conflict(name_a, name_b)` — the canonical rejection function

### Vocab additions
- **Plurals**: berries (strawberries→strawberry, etc.), vegetables
  (tomatoes, onions, …), proteins (sausages, prawns, …)
- **Wine varieties**: chardonnay, cabernet, sauvignon, merlot, pinot,
  shiraz, malbec, riesling, prosecco, …
- **Confectionery / dessert**: peppermint, spearmint, honeycomb, fudge,
  praline, marzipan, nougat
- **`pate`** to one-sided (catches the Ardennes Pâté cluster Darius flagged)

Deliberately *excluded*: `hot`, `sweet`, `spicy` — too broad, trigger
false positives on brand names ("Sweet Freedom") and use-case words
("Hot Chocolate").

## What was deleted (cleanup)

- 7 near-duplicate `scripts/audit_*_code.py` scripts → 1 unified
  `scripts/audit_clusters.py` taking `--input`
- `scripts/filter_final_by_flags.py` (post-LLM filter, no longer needed)
- `legacy_notebooks/` (Jupyter EDA from project start)
- `eda/` (one-time data exploration outputs)
- `src/shopwiser/utils/files_to_txt.py` + `project_dump.txt` (dev tooling)
- `data/outputs/clusters_sample/`, `data/outputs/ml_clusters_sample/`
- 12 stale ensemble intermediate files
- 8 stale completion log CSVs

The pre-cleanup deliverable is preserved at
`data/outputs/backup_pre_v11/ensemble_clusters_final.csv`.

## What's in the hand-back

| Path | Notes |
|---|---|
| `data/outputs/ensemble/ensemble_clusters.csv` | Canonical multi-retailer clusters (current pipeline output) |
| `data/outputs/improvements/ensemble_clusters_with_confidence.csv` | Run `enrich_with_confidence.py` → adds confidence_score / band |
| `data/outputs/improvements/precision_coverage_curve.csv` + `.png` | Operating-point trade-off table |
| `data/outputs/improvements/recovered_pairs.csv` | Residual cross-retailer pairs the prefix-stripping pass would still pick up |
| `data/outputs/fp_analys/fp_candidates.csv` | Heuristic + probe flags per cluster (`export_fp_candidates.py`) |
| `data/outputs/fp_analys/flavour_failures.csv` | Pairwise flavour audit output (`audit_flavour_failures.py`) |
| `data/outputs/improvements/review_sheet.html` + `review_sample.csv` | Clause 4.2 review form, pre-sampled 50 clusters |
| `data/outputs/backup_pre_v11/` | Snapshot of the pre-refactor deliverable |
| `scripts/audit_clusters.py` | Single audit script (replaces the seven `audit_*_code.py` files) |
| `scripts/improvements/COMPARISON_WITH_DARIUS_SPEC.md` | Point-by-point vs Darius's `ShopWiser_Pipeline.pdf` |

## Reproducing every artefact

```bash
# Full pipeline re-run from raw
uv run python main.py normalise
uv run python main.py cluster
uv run python main.py ml-match
uv run python main.py ensemble

# Tests
uv run python main.py test-similarity

# Audits (deterministic, no LLM)
uv run python scripts/audit_clusters.py --input data/outputs/ensemble/ensemble_clusters.csv
uv run python scripts/review_metrics.py --clusters data/outputs/ensemble/ensemble_clusters.csv

# Downstream improvement artefacts
uv run python scripts/improvements/enrich_with_confidence.py
uv run python scripts/improvements/precision_coverage_curve.py
uv run python scripts/improvements/recover_missed_pairs.py
uv run python scripts/improvements/audit_flavour_failures.py

# Clause 4.2 review form
uv run python scripts/improvements/build_review_sheet.py
# After 4 reviewers submit CSVs into data/outputs/improvements/reviews/:
uv run python scripts/improvements/aggregate_review_results.py
```

All deterministic, no GPU, no API calls.

## What we did NOT do (and why)

- **No LLM completion / merge stages.** Kept the pipeline deterministic.
  Reinstating them is a single command (`uv run python main.py
  ensemble-complete`); they would push 4-way count back up at the cost
  of API spend and runtime.
- **No ranker retraining.** The existing `ranker_model.pkl` was reused.
  Retraining with mutual-NN silver labels + the gold set from clause 4.2
  reviews is the next-highest-impact follow-on (see
  `COMPARISON_WITH_DARIUS_SPEC.md`).
- **No bi-encoder fine-tuning.** Off-the-shelf `all-mpnet-base-v2` is
  used. Fine-tuning with triplet loss on the future gold set is the
  Tilburg paper's main lever and would be the largest single
  improvement available to us.
