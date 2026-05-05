# Comparison: ShopWiser pipeline vs Darius's spec + Tilburg paper

A point-by-point read of `ShopWiser_Pipeline.pdf` (Darius's proposed
architecture) and `tilburg.pdf` (multi-modal product matching MSc thesis)
against what we ship in v13.

---

## Architectural alignment

Darius's pipeline is the canonical 3-level matching pattern:

| Darius level | Our module | Status |
|---|---|---|
| **Level A** — bi-encoder + ANN candidate generation | `ml_matcher/retrieval.py` (FAISS over `all-mpnet-base-v2` embeddings) | ✓ aligned |
| **Level B** — cheap gating (own-brand, size, brand, category) | `ml_matcher/main.py` (size gate, product-type gate, hard-conflict gate, brand-token gate) + `rule_matcher/similarity.py` | ✓ aligned, ours is stricter |
| **Level C** — supervised ranker (GBDT or cross-encoder) | `ml_matcher/ranker.py` (LightGBM, 11 engineered features, 150 boost rounds) | ✓ matches Darius's "pragmatic baseline" |
| Silver positives + small gold | `ml_matcher/ranker.py::generate_silver_labels` (Cases A/B/C) | ✓ in place; gold via clause 4.2 review |
| Margin over runner-up at acceptance | `ml_matcher/main.py` (NEW in v13: `MARGIN_THRESHOLD` enforced) | ✓ implemented in v13 |

So the underlying architecture is the same. The differences below are
intentional choices, not gaps.

---

## What we do that Darius's spec doesn't (intentional)

### Multi-supermarket clustering (not just pairwise)
Darius defines `f_θ(i, j)` as a pairwise score and outputs an edge list.
We go further: edges feed into a Kruskal-style union-find with two hard
constraints (one product per supermarket, max cluster size 4) to produce
4-way / 3-way / 2-way clusters directly. This is what the ShopWiser product
needs — show the cheapest "Greek yogurt" across all four retailers in one
row, not a graph of edges the front-end has to assemble.

### Own-brand inclusion (key product divergence)
Darius's spec **excludes** own-brand pairs in "strict identity mode"
(§1.1, scope note). His reasoning: own-brand products are substitutions,
not identities. We **include** them. Why:
- ShopWiser's user value is "what's the cheapest matching basket?", which
  is a substitution question, not an identity question.
- Excluding own-brand would drop ~half the catalogue (Tesco Finest,
  Sainsbury's Taste the Difference, ASDA Extra Special, Morrisons The
  Best, plus value tiers).
- We do enforce **tier compatibility** within own-brand clusters (Finest
  vs Finest, Value vs Value never cross), so the substitution is always
  comparable.

This is the right call for the product, but it should be documented in
the hand-back so reviewers don't apply identity-only metrics to
substitution clusters.

### Hard-conflict vocabulary (richer than Darius's)
Darius's Level B uses size + brand + category as hard gates. We add a
much richer hard-conflict vocabulary in `conflict_tokens.py`:
- `FLAVOR_NAMED_TOKENS` (fruits, herbs, meats, wines, sauces, curries)
- `ONE_SIDED_CONFLICT_TOKENS` (dietary, format, variant markers)
- `MILK_BASE_TOKENS`, `COOKING_STATE_TOKENS`, `PACKAGING_FORMAT_TOKENS`
- `HARD_CONFLICT_NORM` for plural / spelling normalisation

This catches Darius's own §4 examples (Pringles flavour mismatch,
Ardennes Pâté vs Goats Cheese, Pina Colada vs Iced Coffee) before they
ever reach the ranker. The audit on v13 shows only 3 residual hard-conflict
clusters across the whole 13,117-cluster deliverable.

### Sophisticated pack-quantity smart-delta
Supermarkets inconsistently report multipacks: one stores total weight
(1650 ml, pq=NaN), another stores per-unit (275 ml, pq=6). Darius's spec
uses a single `Δsize` ratio. Our `_best_delta_size` in `ml_matcher/features.py`
computes the size delta under three interpretations (per-unit, A-total
vs B-total-from-pack, A-total-from-pack vs B-total) and takes the minimum.
Avoids spuriously rejecting valid multi-pack vs single-pack matches.

---

## What v13 added in response to Darius's spec

### `MARGIN_THRESHOLD` is now actually enforced
The constant `MARGIN_THRESHOLD = 0.04` was defined in `ml_matcher/config.py`
but never read by any code. v13 makes it real: after argmax per
`(anchor, target_supermarket)`, we require the top-1 match score to lead
the runner-up by ≥ 0.04 — Darius's "decisive winner" criterion (§2.6).

Effect on the v13 run: **22,090 of 82,426 above-threshold edges (27%)
were rejected** as ambiguous. Coverage dropped marginally
(38,917 → 38,831 products) but the pipeline is now correctly conservative
on close calls.

### Single source of truth for conflict vocabulary
Previously `clustering/config.py` and `ml_matching/features.py` carried
parallel sets that drifted. v12+ collapses them into
`shopwiser/conflict_tokens.py`. Both pipelines import from one place.

### Plurals normalised at conflict-check time
`HARD_CONFLICT_NORM` now folds berries (raspberries→raspberry),
vegetables (tomatoes→tomato), and protein plurals (sausages→sausage)
to a canonical singular before set-membership lookup. Without this,
"ASDA Raspberries" vs "Sains Strawberries" was matched because
"raspberries" wasn't in `FLAVOR_NAMED_TOKENS`.

---

## What Darius's spec calls for and we still don't do

### Mutual-NN check on silver positives
Darius (§2.4.1, point 4): silver positives require `j ∈ C_sb(i) AND
i ∈ C_sa(j)` with small ranks (top-3). Our `generate_silver_labels`
checks brand/size/category/text-similarity but does NOT verify the
reverse-direction nearest-neighbour rank.

Why we haven't done it: would require regenerating silver labels and
retraining `ranker_model.pkl` — meaningful work that we haven't
justified yet. The current ranker reaches 56% calibrated precision
which is acceptable for v13.

**Suggested follow-on:** add mutual-NN to `generate_silver_labels`,
retrain the LightGBM ranker, re-run pipeline. Expected effect: cleaner
positive labels → tighter ranker → higher precision at same threshold.

### Cross-encoder ranker
Darius (§2.3.1) lists cross-encoder (BERT-style) as a Level C alternative
to GBDT. We use GBDT for speed and interpretability — same call Darius
calls "pragmatic baseline". A cross-encoder could push precision higher
on ambiguous pairs but adds GPU dependency and runtime.

### Gold labels for calibration
Darius (§2.4.3): commission 300–1000 hand-labelled pairs to calibrate the
sigmoid output and prevent self-training drift. We don't have a gold set
yet. The clause 4.2 review (50 clusters × 4 reviewers) will become our
first gold set — once those CSVs come back from Jack/Alex's reviewers we
can use them to calibrate.

---

## What the Tilburg paper adds (mostly out of scope for us)

### Multi-modal (text + image) embeddings with triplet loss
Tilburg fine-tunes Universal Sentence Encoder + DenseNet-201 with triplet
loss + semi-hard negative mining. Their best multi-modal model improves
top-1 accuracy by a few % over text-only. **Not applicable to us:** we
have no image data in the scraped catalogue.

### Multi-label vs binary classification
Tilburg shows multi-label classification (predicting a set of matches
out of top-10 candidates per anchor) outperforms binary pairwise
classification. **Partly applicable:** our Kruskal assembly is effectively
multi-label (an anchor can match 1–3 retailers simultaneously) — we get
the benefit without an explicit multi-label classifier.

### Size features add ~4% to top-1 accuracy
Tilburg confirms what we already do — size is the most important non-text
feature. We use `unit_value` + `pack_quantity` smart-delta (sophisticated
multipack handling), which is strictly more capable than Tilburg's single
size feature.

### Triplet loss for embedding fine-tuning
Tilburg uses triplet loss with semi-hard negatives to fine-tune
embeddings. We use **off-the-shelf** `all-mpnet-base-v2` without
fine-tuning. This is a known precision lever: fine-tuning the embedder
on UK grocery text would tighten the candidate retrieval (Level A) and
let us use a smaller `TOP_K_CANDIDATES`.

**Suggested follow-on:** if we ever build the gold set (above), use it
to mine triplets and fine-tune the bi-encoder. Probably the highest-impact
single change we could make to the pipeline.

---

## Summary

| Topic | Darius's spec | Our v13 |
|---|---|---|
| Bi-encoder + ANN retrieval | Required | ✓ FAISS + mpnet |
| Cheap gating (size/brand/cat) | Required | ✓ stricter |
| Supervised ranker (GBDT) | Pragmatic baseline | ✓ LightGBM |
| Margin over runner-up | Required | ✓ NEW in v13 |
| Silver positive labels | Required | ✓ Cases A/B/C |
| Mutual-NN for silver positives | Required | ✗ open follow-on |
| Gold-set calibration | Recommended | ✗ clause 4.2 will become this |
| Cross-encoder option | Optional | ✗ GBDT is enough |
| Own-brand exclusion | Strict identity | ✗ deliberate divergence (substitution use case) |
| Multi-supermarket clustering | Out of scope | ✓ Kruskal + one-per-SM |
| Hard-conflict vocab (plurals, wine, sauces, etc.) | Out of scope | ✓ conflict_tokens.py |
| Pack-quantity smart-delta | Out of scope | ✓ multi-interpretation min |

The architectures are isomorphic. The deltas are either intentional
product-driven divergences (own-brand inclusion, multi-supermarket
clusters) or richer engineering on top of his spec (vocab, smart-delta).
The two open follow-ons (mutual-NN silver labels, fine-tuned bi-encoder)
both need a gold set first — clause 4.2 will give us one.
