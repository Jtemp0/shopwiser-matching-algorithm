# ShopWiser Matching Pipeline: Independent Review

**Subject**: Review of contractor deliverable `ensemble_clusters_final.csv`

---

## 1. The single most important sentence

The contractor reported "**96.0% of clusters pass all five structural checks**" and described that figure as a precision measure. It is not. It measures whether the pipeline's own rules accept its own output, with rules that quietly let missing data pass. Independent measurement, calibrated against hand-audited samples, puts true match precision at **roughly 50%**, and at **roughly 41% for the 4-way clusters that the contractor highlighted as the headline number**. Every figure in this review is reproducible by running [`reproduce_analysis.py`](./reproduce_analysis.py) against the deliverable.

The contractor did deliver real, useful work: the 2-way clusters, the structured columns, and the overall pipeline shape are largely sound. The issue is validation methodology and the framing of the 4-way numbers as final. Specific recommendations are in §6.

---

## 2. What the contractor built

By examining the columns in the deliverable, the pipeline can be reverse-engineered into seven stages. The contractor did not share source code, so this is inferred from artefacts:

| Stage | What it does | Coverage |
|---|---|---|
| 0. Text canonicalisation | Lowercases, strips punctuation, normalises whitespace. Produces `core_product_name`. | 99.8% |
| 1. Brand recognition | Matches titles against a curated dictionary of ~700 known brands. Produces `known_brand`. | 53.7% |
| 2. Tier tagging | Recognises retailer-specific premium/value tiers (Tesco Finest, ASDA Extra Special, Just Essentials, etc.). Produces `tier_keyword`, `tier_type`. | 5.4% – 25.3% |
| 3. Pack/size extraction | Extracts size in grams or millilitres, mostly inferred from `unit_price ÷ price`. Produces `unit_value`, `unit_type`. | 95.3% |
| 4. Attribute tagging | Tags products with attributes like flavours, dietary, preparation. Produces `attributes_keywords`. | **7.6%** |
| 5. Embedding-based candidate retrieval + LightGBM ranker | The "Stage 1" the contractor described. Produces an intermediate `cluster_id`. | 88.3% |
| 6. Rule-based completion + singleton extension | "Stage 2" + "Stage 3". Stitches together clusters and assigns previously unmatched items. | covers remainder |
| 7. Ensemble merging | Final cluster IDs in `ensemble_cluster_id`. | 100% (12,449 clusters) |

The headline counts match the contractor's email exactly: 6,073 four-way clusters, 3,683 three-way, 2,693 two-way, 40,727 products covered out of the 65k corpus.

The cluster ID column reveals an interesting fact: of the 12,449 final clusters, **8,290 (67%) come straight from the LightGBM ranker (stage 5), and 4,159 (33%) were assembled from multiple sources by the post-hoc stages (6 & 7)**. We can use this to test whether the post-hoc work helped — it does not (see §3.4).

---

## 3. The validation problem

The contractor's "96% pass rate" sounds like precision but isn't. There are four specific reasons.

### 3.1 The structural checks tolerate missing data

The contractor's check 5 ("no mixed branded and own-brand groupings") only flags a cluster if at least one item has `known_brand` filled in *and* another item is own-brand. But the brand layer (§2 stage 1) only recognises 53.7% of items. So when a Heinz product is clustered with three Sainsbury's own-brand products, if Heinz happens to be in the brand dictionary the check fires, but if not — say for a less-popular branded item — the check sees only nulls and quietly passes.

Re-running the contractor's stated checks rigorously, with explicit null-handling, produces these violation counts:

| Check | Definition | Violations |
|---|---|---|
| 1 | One product per supermarket per cluster | 0 |
| 2 | Pack size mismatch ≤ 15% (where sizes are known) | **769** |
| 3 | Recognised brands disagree across cluster | **621** |
| 5 | Branded item in same cluster as own-brand item | **1,066** |

That's **~2,400 cluster-violations** of the contractor's own checks (some clusters fail multiple, so this is an upper bound on failing clusters). The contractor's reported 4% failure rate would imply about **498**.

### 3.2 The checks are circular with the pipeline

The five checks are essentially the same rules used by the pipeline to *build* the clusters. Asking the pipeline to grade itself with the same rules guarantees a high pass rate. A genuine validation needs an independent signal — either human-labelled gold pairs or, at minimum, similarity measures the pipeline didn't use.

We applied a more independent probe ([Part 4 of `reproduce_analysis.py`](./reproduce_analysis.py)). It scores each cluster on five signals:

1. **Character-level cosine similarity** between product names (catches near-identical names with different spacing/punctuation).
2. **Word-level Jaccard overlap** (catches products that share core vocabulary).
3. **Pack size agreement** (sizes within 20% of each other).
4. **At most one distinct brand** across the cluster.
5. **At most one distinct category** across the cluster.

A cluster that passes all five is highly likely to be correct. Results:

| Cluster size | Pass | Fail | % pass | Total |
|---|---|---|---|---|
| 2-way | 2,113 | 580 | **78.5%** | 2,693 |
| 3-way | 1,724 | 1,959 | **46.8%** | 3,683 |
| 4-way | 1,802 | 4,271 | **29.7%** | 6,073 |
| **Overall** | **5,639** | **6,810** | **45.3%** | **12,449** |

### 3.3 Calibrating the probe

Our probe is conservative — it sometimes flags genuinely correct matches (e.g. when retailers use very different product names for the same item). To know what these numbers really mean we hand-audited 20 random clusters: 10 the probe flagged as bad, 10 it flagged as good.

- **Of 10 flagged bad: 8 were genuinely wrong, 2 were borderline-correct** (e.g. "Delamere Whole Milk" vs "Graham's Whole Milk", both 1L — different brands but arguably substitutable, the user's framing).
- **Of 10 flagged good: 9 were genuinely correct, 1 was borderline-wrong** (a flavour mismatch that surface checks couldn't catch).

Combining the probe pass rate with these audit error rates, the calibrated estimate of true precision is:

```
true_precision ≈ probe_pass × 0.90 + probe_fail × 0.20
              ≈ 0.453 × 0.90 + 0.547 × 0.20
              ≈ 51.7%
```

Per cluster size, the calibrated estimates are:
- 2-way clusters: ~75% true precision
- 3-way clusters: ~53% true precision
- **4-way clusters: ~41% true precision**

The 4-way figure is the most consequential because the contractor presented it as the headline. **It is roughly half of what was reported.**

### 3.4 Two further independent signals

Two more checks reinforce the precision concern:

- **Cross-category clusters**: 3,080 clusters (24.7% of the deliverable) span more than one normalised product category. This is a strong signal of an incorrect match — none of the contractor's checks tests it.
- **Adding the post-hoc stages did not help**: comparing the 8,290 "pure stage 1" clusters vs the 4,159 "post-hoc assembled" clusters on the independent probe, both score ~30% pass rate for 4-way clusters. The stage-2 and stage-3 work cost about a thousand 4-way clusters of headline coverage (the "earlier 7,000+ figure" the contractor mentioned) without measurably improving quality.

---

## 4. Concrete examples

The numbers above are easier to act on with concrete examples. Here are randomly-sampled 4-way clusters where the probe flagged the names as too dissimilar (run Part 5 of the script for more):

> **Cluster 41** &nbsp; *Same product across four retailers?*
> - [ASDA] £4.50 — Starbucks Single Origin Colombia Medium Roast Ground Coffee
> - [Sains] £4.50 — Starbucks Colombia Single-Origin Medium Ground 100% Arabica Coffee Bag 200g
> - [Tesco] £4.50 — **Tesco Finest Sunset Hour Ground Coffee 227G**

> **Cluster 6933** &nbsp; *Same product?*
> - [ASDA] £2.50 — ASDA To Share 6 Vegetable Spring Rolls 216g
> - [Tesco] £2.25 — Tesco 6 Vegetable Spring Rolls 216G
> - [Sains] £1.55 — **Trex Vegetable Fat 250g**

> **Cluster 63** &nbsp; *Same product?*
> - [ASDA] — Jimmy's Iced Coffee Original
> - [Sains] — Jimmy's Iced Coffee Original 275ml
> - [Tesco] — Jimmy's Iced Coffee Original 275Ml
> - [Morrisons] — **Morrisons Pina Colada**

> **Cluster 4023** &nbsp; *Same product?*
> - [ASDA] £2.00 — ASDA Spreadable Goats' Cheese 150g
> - [Morrisons] £2.39 — Morrisons Spreadable Goats Cheese
> - [Sains] £1.00 — **Sainsbury's Ardennes Spreadable Pâté 175g**
> - [Tesco] £3.35 — Tesco Finest Kidderton Ash Goats Cheese 150G

> **Cluster 5445** &nbsp; *Same product?*
> - [ASDA] £1.85 — Pringles Sour Cream & Onion Sharing Crisps
> - [Morrisons] £1.85 — **Pringles Cheese & Onion Sharing Crisps**
> - [Sains] £2.25 — Pringles Sour Cream & Onion Sharing Crisps 185g
> - [Tesco] £1.50 — Pringles Sour Cream & Onion Sharing Crisps 185g

These were all reported as passing the contractor's checks. The crisps example (5445) is particularly diagnostic: same brand, same pack size, same category — the only difference is *flavour*, and the attribute layer (which was supposed to catch this) was filled for only 7.6% of items, so the check has no signal.

---

## 5. Specific claims, item by item

A claim-by-claim review of the contractor's email:

| Claim | Verdict | Evidence |
|---|---|---|
| "The final pipeline produces 12,449 matched product groups" | **True** | Counts match exactly. |
| "6,073 four-way, 3,683 three-way, 2,693 two-way" | **True** | Counts match exactly. |
| "Covering 40,727 out of 65,023 total scraped products (62.6% coverage)" | **True** | The deliverable contains 40,727 rows. |
| "96.0% of clusters pass all five checks" | **Misleading** | True under their definitions, but those definitions tolerate nulls (§3.1). Calibrated true precision: ~50%. |
| "Five structural checks: one product per supermarket, no size mismatch >15%, no brand conflicts, no flavour or variant token clashes, no mixed branded/own-brand" | **Partly correct as stated; weak in implementation** | The flavour-token check should have caught Cluster 5445 above. It didn't, because 92% of items have no machine-readable flavour. |
| "An earlier 7,000+ figure reflected a pipeline run before additional checks were applied" | **True but doesn't help** | The post-hoc filtering reduced 4-way coverage by ~1,000 clusters with no measurable precision gain (§3.4). |
| "6,073 is the ceiling for a deterministic, rule-based system on this catalogue" | **False** | Sampling unmatched items shows ~10% of them have a strong cross-retailer match the deliverable missed. Concrete examples below. |
| "Roughly 5,000 of the remaining singleton products simply have no cross-retailer equivalent" | **Overstated** | Of 500 randomly-sampled unmatched grocery items, 52 (10.4%) have a similarity-≥0.7 match in another retailer. Extrapolated to the full unmatched grocery pool that's ~1,150 missed pairs in just the sample corpus, more in the full 65k. |
| "A further several thousand differ in name phrasing... that a rule-based system cannot safely resolve without risking false matches" | **Partly true, but misframed** | The right framing is precision-vs-coverage, not "ceiling". Different operating points reach different counts. |
| "Around 2,000 product names are truncated" | **Inflated** | 899 in the deliverable; the rest may be in the unmatched pool, but the figure as stated overstates the issue. |
| "All pipeline stages are reproducible Python modules (which I need to document a bit further before handover...)" | **Yellow flag** | Insist on documentation and a runnable end-to-end pipeline before sign-off. |

### 5.1 Examples of cross-retailer matches the deliverable missed

These were found by stripping retailer prefixes from titles and searching for high-similarity matches among unmatched items. None required ML — just one rule the deliverable missed (consistent retailer-prefix stripping):

| Retailer A | Retailer B | Similarity |
|---|---|---|
| Morrisons Organic Brown Onions | ASDA Organic Brown Onions | 1.00 |
| ASDA Straight Cut Chips | Morrisons Straight Cut Chips | 1.00 |
| Morrisons Chicken Caesar Wrap | Tesco Chicken Caesar Wrap | 1.00 |
| Sainsbury's Southern Fried Chicken Wrap | Morrisons Southern Fried Chicken Wrap | 1.00 |
| Sainsbury's TtD Alla Genovese Pesto 190g | ASDA Extra Special Pesto Alla Genovese 190g | 1.00 |
| Morrisons Waffle Cones | Tesco 10 Waffle Cones | 0.96 |
| Morrisons Seedless Red Grapes | Tesco Finest Red Grapes Seedless 400G | 0.95 |
| ASDA Chopped Mixed Nuts 150g | Morrisons Chopped Mixed Nuts | 0.93 |
| Tesco Finest Orzo Pasta 500G | ASDA Orzo Pasta | 0.92 |
| ASDA Extra Special Olive & Rosemary Sourdough 500g | Tesco Finest Green Olive & Rosemary Sourdough 400G | 0.89 |
| Morrisons Sage & Onion Stuffing | JUST ESSENTIALS by ASDA Sage and Onion Stuffing Mix | 0.90 |

These are all genuinely the same or near-identical products that the deliverable left as singletons.

---

## 6. What to do with the deliverable

A simple traffic-light recommendation:

### Keep and use

- **The 2-way clusters** (2,693 clusters): calibrated precision ~75%. Workable in production with a confidence band attached.
- **The structured intermediate columns**: `core_product_name`, `normalized_name`, `unit_value`/`unit_type`, `tier_keyword`, `pack_quantity`, `cat_norm`. These are useful intermediate artefacts regardless of what happens to the clusters themselves.

### Hold for review

- **The 3-way clusters** (3,683 clusters): calibrated precision ~53%. Roughly half are correct; without a second-pass verification step, treating these as final risks user-visible errors. Either re-rank with a stricter scorer or surface confidence in the UI (see §7).

### Do not use as-is

- **The 4-way clusters** (6,073 clusters): calibrated precision ~41%. The contractor's headline number. Three out of every five are wrong, in ways that will produce visibly incorrect cheapest-supermarket recommendations.

### Three concrete asks for the contractor before sign-off

1. **The validation script and its gold set.** If clause 4.2's Q1/Q2/Q3 are the same five structural checks, the validation is circular and the 96% number doesn't measure what was claimed. Ask for an independent gold set with adversarial samples (mid-confidence cases, same-brand-different-flavour pairs, premium-vs-value tier conflicts).

2. **A precision-vs-coverage curve at multiple operating points**, not a single point estimate. The right shape of answer is "X clusters at 90% precision, Y at 80%, Z at 70%", which lets you choose the operating point downstream.

3. **Per-cluster confidence scores** in the deliverable, not just pass/fail. These are essential for any uncertainty-aware UI and also let you quickly re-bucket without re-running the pipeline.

These three asks could help make the deliverable usable even before any rework.

---

## 7. The single biggest methodological point: confidence is the missing primitive

The deepest issue with the deliverable isn't any specific bug — it's that the pipeline produces binary decisions (in-cluster / not-in-cluster) when the underlying problem is fundamentally probabilistic. Some products are genuinely the same; some are clearly different; many are in between. A pipeline that hides that uncertainty in a binary label loses information the user needs to make a sensible choice.

The right primitive is a **calibrated confidence score per cluster**, surfaced explicitly in the user interface. "Calibrated" means: when the score says 70%, the cluster is correct 70% of the time on a held-out test set. This is what lets the product say honest things like "Tesco saves you £4.20 on this basket, but two of your six items are matched with low confidence, please verify those items before you commit."

Confidence is also what makes the pipeline actually improvable. With a confidence score per cluster you can:
- Filter the deliverable to only the high-confidence subset (= ship now, with smaller catalogue but trusted prices)
- Route low-confidence clusters to human review (= the engine for an active-learning loop)
- Show the user the precision/coverage tradeoff and let them choose

The accompanying React POC ([`shopwiser_uncertainty.jsx`](./shopwiser_uncertainty.jsx)) demonstrates what this looks like end-to-end:

- Every product carries a confidence chip and a meter.
- The basket-comparison view shows an overall trust score and warns when low-confidence items are skewing the cheapest-store verdict.
- A "match-quality filter" lets the user choose how strict they want to be, directly visualising the precision-coverage tradeoff.
- Tapping any product reveals exactly *why* the confidence is what it is (size mismatch? brand conflict? flavour token clash?).
- Each retailer's contribution to the basket total is split into "trusted" and "uncertain" portions.

The seeded basket includes one very-low-confidence item on purpose so the warning behaviour is visible immediately. Open `Compare` to see it.

The POC is fed by the same per-cluster scores produced by `reproduce_analysis.py`, so the connection between "what we measured" and "what the user sees" is direct.

---

## 8. Reproducibility

Every numerical claim can be reproduced by:

```bash
# Place the input files alongside the script:
#   ./data.csv                       (the original scraped corpus)
#   ./ensemble_clusters_final.csv    (the contractor's deliverable)

pip install pandas numpy scikit-learn
python reproduce_analysis.py
```

The script runs in 2–3 minutes on a laptop and produces:

- Console output with seven labelled sections, each mapping to a section of this review
- `output/cluster_stats.csv` — per-cluster quality scores (used in §3.3)
- `output/missed_matches.csv` — examples of cross-retailer pairs the deliverable missed (used in §5.1)

The script is plain pandas/numpy/scikit-learn — no exotic dependencies, no GPU, no external API calls. It reads as a structured set of probes you can run, modify and re-run without ML expertise.

---

## 9. Appendix: package contents

This review hand-off contains:

- **`REVIEW.md`** &nbsp; (this document)
- **`reproduce_analysis.py`** &nbsp; reproducible analysis script — every claim above is grounded in its output
- **`shopwiser_uncertainty.jsx`** &nbsp; single-file React app demonstrating how confidence-aware matching surfaces in the user product

The React POC is a single-file JSX artefact — drop it into any React + Tailwind project (or paste it into the Claude artifact viewer) to see the user experience directly. It uses 62 real clusters from the deliverable, hand-selected to span the full confidence range so the uncertainty UI is visible from the start.
