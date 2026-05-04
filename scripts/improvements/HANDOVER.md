# ShopWiser Matching Pipeline — Handover (post-review revisions)

This is a focused note describing the changes made in response to Darius's
review and how to reproduce / use them. The original deliverable
(`ensemble_clusters_final.csv`) is **unchanged**; everything here is additive
and lives under `data/outputs/improvements/`.

---

## What's new

| File | What it is |
|---|---|
| `data/outputs/improvements/ensemble_clusters_final_v2.csv` | The original deliverable + a per-cluster `confidence_score` (0–1) and `confidence_band` (high / medium / low). |
| `data/outputs/improvements/precision_coverage_curve.csv` + `.png` | Cluster counts and probe-pass rate at confidence thresholds 0.30 → 0.90. The shape of answer for choosing where the MVP cuts. |
| `data/outputs/improvements/recovered_pairs.csv` | 508 obvious cross-retailer pairs the original deliverable missed (e.g. ASDA vs Morrisons "Organic Brown Onions"). Catches the §5.1 cases in Darius's review. |
| `data/outputs/improvements/flavour_failures.csv` | 292 clusters in the existing deliverable that the new flavour-vocab expansion would now reject (carbonara vs bolognese, hoisin vs teriyaki, decaf vs non-decaf, etc.). Quantifies the precision lift of the v11 vocab. |
| `data/outputs/improvements/review_sheet.html` | The clause 4.2 review form. Open in a browser, hand to a reviewer, they answer the three Yes/No questions per cluster and a CSV downloads on submit. |
| `data/outputs/improvements/review_sample.csv` | The 50 clusters sampled into the form, in machine-readable form. |

---

## How to run the clause 4.2 validation

1. Generate the sample (already done — re-run with a different seed if needed):
   ```bash
   uv run python scripts/improvements/build_review_sheet.py --seed 42 --n 50
   ```
2. Open `data/outputs/improvements/review_sheet.html` in any browser.
3. Send to four unaffiliated reviewers. Each fills the form and submits — the
   browser downloads `shopwiser_review_<reviewer>.csv` to their machine.
4. Drop all four CSVs into `data/outputs/improvements/reviews/`.
5. Aggregate:
   ```bash
   uv run python scripts/improvements/aggregate_review_results.py
   ```
   The script reports the contractual pass rate (clusters that ≥3 of 4
   reviewers marked passing) against the 90% threshold in clause 4.4.

---

## What changed in the pipeline source

### Expanded flavour / variant vocabulary
File: `src/shopwiser/clustering/config.py`

Added named-flavour tokens covering UK savoury snacks, sauces, condiments,
curries and cheese types (Pringles flavour mismatches, hoisin vs teriyaki,
carbonara vs bolognese, etc.). Added `sour`, `salted`, `unsalted`, `smoky`
to one-sided conflict tokens.

Deliberately **excluded**: `hot`, `sweet`, `spicy` — too broad in UK
grocery naming (e.g. "Hot Chocolate", "Sweet Freedom" brand) and trigger
false positives.

Quantified impact on the existing deliverable: 292 clusters carry pair-wise
flavour conflicts the new vocab catches but the previous run did not.
These are the next pipeline regeneration's expected precision gain.

### New retailer-prefix recovery
File: `scripts/improvements/recover_missed_pairs.py`

Strips ~18 retailer / sub-brand prefixes (`ASDA Extra Special`, `Just
Essentials by ASDA`, `Tesco Finest`, `Morrisons The Best`, `Sainsbury's
Taste the Difference`, plain retailer names) and finds high-similarity
cross-retailer pairs among the singletons. Output is candidate pairs to
feed back into the next pipeline run as seed matches.

---

## Reproducing every artefact

```bash
# 1. Per-cluster confidence column
uv run python scripts/improvements/enrich_with_confidence.py

# 2. Precision-vs-coverage curve + PNG
uv run python scripts/improvements/precision_coverage_curve.py

# 3. Recovered missed cross-retailer pairs
uv run python scripts/improvements/recover_missed_pairs.py

# 4. Flavour-vocab failure audit (292 clusters affected)
uv run python scripts/improvements/audit_flavour_failures.py

# 5. Build the clause 4.2 review form
uv run python scripts/improvements/build_review_sheet.py

# 6. After 4 reviewers submit their CSVs:
uv run python scripts/improvements/aggregate_review_results.py
```

All scripts are pandas / numpy / scikit-learn only. No GPU, no API calls.

---

## Reading the precision-vs-coverage table

From `precision_coverage_curve.csv`:

| Threshold | Clusters kept | 4-way | 3-way | 2-way | Probe pass % |
|---|---|---|---|---|---|
| 0.40 | 11,338 (91%) | 5,200 | 3,446 | 2,692 | 49.7% |
| 0.55 | 9,081 (73%)  | 3,618 | 2,779 | 2,684 | 62.1% |
| 0.70 | 7,142 (57%)  | 2,397 | 2,138 | 2,607 | 74.4% |
| 0.85 | 4,942 (40%)  | 1,341 | 1,359 | 2,242 | 78.2% |

Interpretation: the deliverable is not "ship all or ship nothing". Picking a
threshold lets you trade catalogue size for cluster cleanliness. For an
MVP cheapest-basket experience, the **0.70–0.75 band** is a reasonable
launch point (~7,000 clusters, ~75% probe pass), with the lower-confidence
remainder routed to a verification queue.

The probe pass rate is the same TF-IDF + token-overlap probe Darius used,
applied per threshold band. It is a deterministic proxy, not a substitute
for the clause 4.2 panel review.

---

## What this does NOT do

- **Re-run the full pipeline.** Regenerating `ensemble_clusters_final.csv`
  with the new vocab + recovered pairs is the next step but it requires a
  full retrieval + ranking pass and was deliberately scoped out of this
  hand-back. The audits above quantify the expected delta so the call to
  re-run is informed.
- **Replace the contractual validation.** The structural probe and the
  calibrated bands are engineering tools; the clause 4.2 panel review is
  the contractual measure and should run alongside, not instead of, them.
