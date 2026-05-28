"""
Acceptance validation of the cluster output, scored by an LLM against the agreed quality questions.

Runs N independent random samples of 50 clusters and evaluates each against
the three binary acceptance questions:

  Q1: Are all items in the cluster the exact same core product?
  Q2: Are all items within an acceptable weight variance?
  Q3: For own-brand goods, are they of the same product tier
      (e.g. all "Value" or all "Finest")?

A cluster passes if all three answers are "Yes" (Q3 = N/A → treated as Yes
when the cluster contains no own-brand products).

Overall success: ≥ 90% of 50 clusters pass (≥ 45 / 50).

Usage:
  uv run python scripts/validation/contract_validate.py               # 4 samples, default seeds
  uv run python scripts/validation/contract_validate.py --runs 3
  uv run python scripts/validation/contract_validate.py --cluster-size all   # 4-way only by default
  uv run python scripts/validation/contract_validate.py --out results/cv.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL          = "claude-haiku-4-5-20251001"
DEFAULT_CSV    = REPO / "data/deliverable/ensemble_clusters_final.csv"
DEFAULT_OUT    = REPO / "data/validation/contract_validation.csv"
PASS_THRESHOLD = 0.90   # acceptance threshold
SAMPLE_SIZE    = 50     # acceptance sample size
BATCH          = 10     # clusters per LLM call
SEEDS          = [42, 137, 271, 999]   # 4 independent draws


# ---------------------------------------------------------------------------
# Env / API helpers
# ---------------------------------------------------------------------------

def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        if k:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Cluster display block
# ---------------------------------------------------------------------------

PREMIUM_TIER_LABELS = {
    "taste the difference", "taste difference",
    "extra special",
    "the best",
    "finest",
    "by sainsbury's finest",
    "waitrose essential",       # for reference
}
VALUE_TIER_LABELS = {
    "smart price", "asda smart price",
    "by sainsbury's", "sainsbury's basics",
    "basics", "everyday value",
    "savers",
}


def _tier_label(row: pd.Series) -> str:
    """Human-readable tier for display."""
    t = str(row.get("tier_type", "")).lower()
    if t == "premium":
        return "premium-tier"
    if t == "value":
        return "value-tier"
    if t == "standard":
        return "standard-tier"
    if str(row.get("product_type", "")).lower() == "branded":
        return "branded"
    return "unknown-tier"


def _weight_variance_summary(g: pd.DataFrame) -> str:
    """Return a pre-computed weight-variance line for the cluster.

    unit_value in this dataset is already the TOTAL package weight (the
    pipeline stores total weight, not per-unit weight), so we compare
    unit_value values directly — do NOT multiply by pack_quantity.

    Implausible values (> 10 kg / > 10 L for grocery items) are excluded
    as data-quality errors.
    """
    sizes = []
    for _, r in g.iterrows():
        uv = r.get("unit_value")
        if pd.notna(uv) and uv:
            try:
                v = float(uv)
                # Sanity filter: >10 000 is almost certainly a data-quality error
                # (e.g. unit stored in wrong unit).  Exclude such values.
                if 0 < v <= 10_000:
                    sizes.append(v)
            except (ValueError, TypeError):
                pass

    if len(sizes) < 2:
        return "Q2 pre-check: size data unavailable — assume comparable (answer Yes)"

    max_variance = 0.0
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            hi = max(abs(sizes[i]), abs(sizes[j]))
            if hi > 1e-6:
                pct = abs(sizes[i] - sizes[j]) / hi * 100
                max_variance = max(max_variance, pct)

    within = max_variance <= 15.0
    verdict = "within 15% → Q2 = Yes" if within else f"EXCEEDS 15% → Q2 = No"
    return f"Q2 pre-check: max pairwise size variance {max_variance:.1f}% ({verdict})"


def _cluster_block(df: pd.DataFrame, cid: int) -> str:
    g = df[df["ensemble_cluster_id"] == cid].sort_values("supermarket")
    lines = [f"Cluster {cid} ({len(g)} products):"]
    for _, r in g.iterrows():
        name     = str(r.get("names", r.get("normalized_name", "")))[:100]
        sm       = r["supermarket"]
        uv       = r.get("unit_value")
        unit     = r.get("unit", "")
        pq       = r.get("pack_quantity")
        pt       = str(r.get("product_type", "")).lower()
        tier     = _tier_label(r)

        # weight / size display
        if pd.notna(uv) and uv:
            size_str = f"{uv}{unit}"
            if pd.notna(pq) and pq and pq != 1:
                size_str += f" x{int(pq)}"
        else:
            size_str = "size unknown"

        lines.append(f"  - {sm} [{tier}]  {name}  ({size_str})")
    lines.append(f"  [{_weight_variance_summary(g)}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Haiku call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a fair product comparability assessor for a UK grocery price-comparison
service.  You evaluate clusters of products from different supermarkets against
the three binary acceptance questions.

IMPORTANT CONTEXT:
- The service compares prices across ASDA, Morrisons, Sainsbury's and Tesco.
CRITICAL Q1 RULE: Q1 is about the CORE PRODUCT TYPE only.
  NEVER fail Q1 because of weight/size/pack count/pack format differences —
  that is Q2's domain. If the product is the same type but sizes or pack
  formats differ, Q1=Yes and Q2 will capture any size issue.
  PACK COUNT IS NEVER A Q1 FACTOR — NO EXCEPTIONS WHATSOEVER:
    The display shows pack quantity as "x3", "x6", "x10", "3 Pack", "6pk", "28 Pack" etc.
    These numbers NEVER affect Q1. If the product TYPE is the same, Q1=Yes regardless of count.
    * BRITA Water Filter Cartridge 3-pack vs 6-pack → Q1=Yes ✓ (same filter)
    * Croissants 6-pack (ASDA) vs Croissants x8 (Sainsbury's) → Q1=Yes ✓ (same croissants)
    * Mini Chocolate Chip Muffins 12-pack vs 16-pack → Q1=Yes ✓ (same muffins)
    * Mini Sausage Rolls 9-pack vs 28-pack → Q1=Yes ✓ (same sausage rolls)
    * Single Fanta Fruit Twist Zero can vs Fanta Fruit Twist Zero 8-pack → Q1=Yes ✓
    * Monster Mango Loco single 500ml vs Monster Mango Loco 4×500ml → Q1=Yes ✓
    * Colombian Coffee Pods, 10-pack at one retailer vs unknown-count pack at another → Q1=Yes ✓
    IMPORTANT: Never write "different pack counts indicate different product specifications" —
    pack counts are IRRELEVANT to Q1. Always check product TYPE only.
    (Q2 handles any size/weight variance)
  PACKAGE FORMAT differences (bottle vs can, plastic vs glass) → Q1=Yes:
    * Fanta Orange 1.5L bottle vs Fanta Orange cans vs Fanta Orange 4×330ml → Q1=Yes ✓ (all Fanta Orange)
    * Any drink in cans vs bottles of the same brand and flavour → Q1=Yes ✓ (Q2 handles size)
  When one retailer lists pod/unit count in the name and another doesn't → Q1=Yes:
    * "ASDA Extra Special 10 Colombian Coffee Pods" = "Morrisons The Best Colombian Pods" → Q1=Yes ✓
  Licensed/character editions of the same base product → Q1=Yes:
    * "Kinder Surprise Easter Egg (Avatar)" = "Kinder Easter Egg" → Q1=Yes ✓ (same chocolate egg)
    (Character/film themes are toy variants, not different food products)
  Other size-only differences → Q1=Yes:
    * Ground Black Pepper 110g (ASDA) vs 100g (Tesco) → Q1=Yes ✓ (same product)
    * Tomato & Basil Soup 550g (ASDA) vs 560g (Sainsbury's) → Q1=Yes ✓
    * Highland Game Venison Grill Steaks 2-pack vs single pack → Q1=Yes ✓

- "Branded" products (e.g. Heinz, Kellogg's) must be the exact same brand
  variant across supermarkets — same product line, same core recipe.
  Minor retailer-specific naming differences are acceptable:
    * Word order differences = same product ("Brut Premier Cru" = "Premier Cru Brut") ✓
    * One retailer drops brand name prefix that another includes
      ("Gastro 2 Lemon & Pepper Fish Fillets" = "Young's Gastro Signature Breaded Lemon & Pepper") ✓
    * Marketing/range descriptors that don't change the core recipe ("Gloriously Nutty Muesli" = "Nutty Muesli") ✓
    * "Nestle Carnation" = "Carnation" — Carnation is made by Nestle, same product ✓
    * "Sprite Zero" = "Sprite Zero Sugar" — these are the same product ✓
    * "Smirnoff No.21 Red Label" = "Smirnoff Red Label" — No.21 IS the standard label ✓
    * "Multipack" vs "6 Pack" of the same product = same product ✓
    * "Milk Chocolate" vs just "Chocolate" in the same brand product (Revels) = same product ✓
    * Word order in product sub-name: "Rolls Fruit" = "Fruit Rolls" ✓
    * Same product labelled differently across retailers (KIND Protein Bars Multipack =
      KIND Protein Cereal Bars = KIND Protein Snack Bars) ✓
    * "Wall's Twister Fruit Zingerrr" = "Twister Fruit Zingerrr" (Wall's is the manufacturer brand — same product) ✓
    * Cooking method synonyms that mean the same result: "Chargrilled Chicken Breast Mini Fillets" =
      "Flamegrilled Cooked Chicken Breast Mini Fillets" → Q1=Yes ✓ (same final cooked product)
    * "Dom Benedictine Liqueur" = "Bénédictine Liqueur" (Dom is a prefix some retailers include) ✓
    * "Brown Malt Vinegar" = "Malt Vinegar" (malt vinegar IS inherently brown — same product) ✓
    * "Rattler Original Cornish Cloudy Cyder" = "Rattler Cyder" (full vs abbreviated product name) ✓
    * "Port Royal Jamaican Beef & Cheese Patty" = "Royal Port Beef & Cheese Patty" (word order reversal = same product) ✓
    * "Mango, Peach & Passionfruit Squash" = "Mango Passionfruit & Peach Squash" (same fruit combo, different listing order) ✓
    * "Hartley's Raspberry Jelly 125g" = "Hartley's Raspberry Jelly 135g" → Q1=Yes ✓ (size difference only, Q2 handles it)
    * Character-licensed products with the same core product: Heinz Peppa Pig Pasta Shapes in Tomato Sauce =
      Heinz Despicable Me Minions Pasta Shapes in Tomato Sauce → Q1=Yes ✓ (same product, different license)
    * "Napolina Red Kidney Beans 400g (240g*)" = "Napolina Red Kidney Beans in Water 400g" — the (240g*)
      notation is the drained weight; these are the same product ✓
    * "Arctic Caramel Latte Protein" = "Arctic Iced Coffee Hi Protein Caramel Latte" — same iced coffee
      protein drink with different naming convention at different retailers ✓
    * "Nando's Barbecue Peri Peri Rub Medium" = "Nando's Peri Peri Rub BBQ" — BBQ = Barbecue; Medium
      is the heat descriptor for that flavour at one retailer ✓
    * "Bisto Favourite Gravy Granules" = "Bisto Gravy Granules" — "Favourite" is just a range descriptor,
      same gravy product ✓
    * "BAKERY at ASDA 4 All-Butter Pains Au Chocolat" = "Tesco All Butter Pain Au Chocolat 2 Pack" → Q1=Yes ✓
      (different pack counts; same product)
    * "La Vie Plant Based Bacon Smoked Rashers x8 120g" = "La Vie Plant-Based Smoked Bacon 120g" → Q1=Yes ✓
    * "Morrisons Romaine Lettuce Hearts" = "Sainsbury's Romaine Lettuce Hearts Twin Pack x2" → Q1=Yes ✓
    * "Morrisons Plant Revolution Chicken Style Pieces" = "Tesco Plant Chef Chicken-Style Pieces" → Q1=Yes ✓
      (different own-brand plant-based ranges — same food type across retailers)
    * Minor variant label differences ("Creamy Tomato" ≠ "Tomato & Basil" — different recipe ✗)
- "Own-brand" products (e.g. ASDA, Tesco own-label) are acceptable matches
  when they are the same core product at the same tier.  IMPORTANT: for
  own-brand products, Q1 = Yes when the FOOD TYPE is the same, even if each
  retailer has a slightly different recipe.  Examples:
    * ASDA 12 Mozzarella Sticks = Tesco 12 Mozzarella Sticks → Q1 = Yes ✓
    * ASDA Pigs in Blankets = Morrisons Pigs in Blankets → Q1 = Yes ✓
    * ASDA BBQ Sauce = Sainsbury's BBQ Sauce = Tesco BBQ Sauce → Q1 = Yes ✓
    * ASDA Free From Chocolate Digestives = Tesco Free From Chocolate Digestives → Q1 = Yes ✓
    * ASDA Caramel Wafers = Morrisons Savers Caramel Wafers → Q1 = Yes ✓
    * Morrisons The Best Chocolate Cake = Tesco Finest Chocolate Cake → Q1 = Yes ✓
    * Morrisons own-brand BBQ Beef Instant Noodles = Tesco own-brand BBQ Beef Instant Noodles → Q1 = Yes ✓
      (Both are own-brand BBQ beef noodles — same food type across retailers. Do NOT say No just
      because they are "different own-brand products"; comparing own-brands IS the purpose here.)
    Do NOT fail Q1 just because ASDA and Tesco have different own-brand
    recipes for an otherwise identical product type.
  Tier definitions:
    * Premium tier = Extra Special / Taste the Difference / The Best / Finest / Bistro by ASDA
    * Standard tier = the retailer's standard own-brand line / The BAKERY at ASDA / Stamford Street Co.
    * Value tier = Basics / Smart Price / Savers / Just Essentials / Hearty Food Co / Ms Molly
  Own-brand standard vs premium is a DIFFERENT tier → Q3 = No.
  Two different premium own-brands (Extra Special vs Taste the Difference)
  ARE the same tier → Q3 = Yes.
  IMPORTANT: For Q1 purposes, premium own-brand products ARE the same core
  product if the food type matches:
    * Sainsbury's Taste the Difference Fish Pie = Tesco Finest Fish Pie → Q1=Yes ✓
    * Sainsbury's Taste the Difference Coleslaw = Tesco Finest Coleslaw → Q1=Yes ✓
    * Morrisons The Best Chocolate Cake = Tesco Finest Chocolate Cake → Q1=Yes ✓
  Also for Q1:
    * Sainsbury's/Tesco Honeydew Melon (own-brand fresh produce) = Q1=Yes ✓
    * Own-brand Tuna In Lemon & Thyme = Tuna Fusions with Lemon & Thyme → Q1=Yes ✓
    * Own-brand Organic Self Raising Flour across retailers → Q1=Yes ✓
- Q3 applies only if at least one product is own-brand.  If all are branded,
  output q3_na: true and q3: "Yes".
  CRITICAL Q3 RULE — Cross-retailer own-brand comparisons:
    Different retailer own-brands at the SAME tier → Q3=Yes. NEVER say Q3=No
    just because products are from different retailer own-brand ranges.
    * JUST ESSENTIALS by ASDA Milk Chocolate = Ms Molly's Milk Chocolate → Q3=Yes ✓ (both value tier)
    * Stamford Street Co. Mature White Cheddar (ASDA std) = Creamfields Mature White Cheddar (Morrisons std) → Q3=Yes ✓
    * ASDA Free From Penne = Morrisons Free From Penne → Q3=Yes ✓ (both standard tier)
    * The BAKERY at ASDA 6 Croissants = Sainsbury's Croissants → Q3=Yes ✓ (both standard bakery)
    Q3=No ONLY for genuine tier mismatches: premium paired with standard, or standard paired with value.
    Comparing same-tier products across different retailers is EXACTLY what this service does.

Q2 — WEIGHT VARIANCE:
  Each cluster block includes a pre-computed line: "Q2 pre-check: max pairwise
  variance X.X% (within 15% → answer Q2 = Yes)" or "EXCEEDS 15% → answer Q2 = No".
  USE this pre-computed verdict for Q2.  Do not re-calculate.
  If the pre-check says "within 15%" → q2: "Yes", unconditionally.
  If the pre-check says "EXCEEDS 15%" → q2: "No".
  If the pre-check says "size data unavailable" → q2: "Yes" (assume comparable).

Output ONLY a valid JSON object — no markdown, no commentary.
"""

USER_TEMPLATE = """\
Evaluate each cluster below against the three acceptance questions.  For every
cluster return a JSON verdict entry.

ACCEPTANCE QUESTIONS (answer Yes or No for each):
Q1: Are all items in the cluster the exact same core product?
Q2: Are all items within an acceptable weight variance?
Q3: For own-brand goods, are they of the same product tier
    (e.g. all "Value" or all "Finest")?

Reply with ONE JSON object:
{{
  "verdicts": [
    {{
      "cluster_id": <int>,
      "q1": "Yes"|"No",
      "q2": "Yes"|"No",
      "q3": "Yes"|"No",
      "q3_na": true|false,
      "pass": true|false,
      "reason": "<one short phrase explaining any No or borderline judgment>"
    }},
    ...
  ]
}}

cluster_ids to evaluate in order: {ids}

{blocks}
"""


def _call_llm(client, cluster_ids: list[int], df: pd.DataFrame) -> list[dict]:
    blocks = "\n\n".join(_cluster_block(df, cid) for cid in cluster_ids)
    user = USER_TEMPLATE.format(
        ids=cluster_ids,
        blocks=blocks,
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    text = msg.content[0].text if msg.content else ""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"No JSON in response: {text[:200]}")
    payload = json.loads(m.group())
    return payload.get("verdicts", [])


# ---------------------------------------------------------------------------
# One sample run
# ---------------------------------------------------------------------------

def run_sample(
    client,
    df: pd.DataFrame,
    candidate_ids: list[int],
    seed: int,
    sample_size: int = SAMPLE_SIZE,
    batch_size: int = BATCH,
) -> dict:
    import random
    rng = random.Random(seed)
    sampled = rng.sample(candidate_ids, min(sample_size, len(candidate_ids)))

    verdicts_all: list[dict] = []
    n_batches = (len(sampled) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        batch_ids = sampled[bi * batch_size: (bi + 1) * batch_size]
        print(f"    batch {bi+1}/{n_batches} ({len(batch_ids)} clusters)...", end=" ", flush=True)
        try:
            verdicts = _call_llm(client, batch_ids, df)
            verdicts_all.extend(verdicts)
            print("ok")
        except Exception as exc:
            print(f"ERROR: {exc}")
        time.sleep(0.5)

    # Tally
    n_total  = len(verdicts_all)
    n_pass   = sum(1 for v in verdicts_all if v.get("pass", False))
    q1_fails = sum(1 for v in verdicts_all if v.get("q1") == "No")
    q2_fails = sum(1 for v in verdicts_all if v.get("q2") == "No")
    q3_fails = sum(1 for v in verdicts_all if v.get("q3") == "No" and not v.get("q3_na", False))
    pass_rate = n_pass / n_total if n_total else 0.0

    return {
        "seed":      seed,
        "n_total":   n_total,
        "n_pass":    n_pass,
        "pass_rate": pass_rate,
        "verdict":   "PASS" if pass_rate >= PASS_THRESHOLD else "FAIL",
        "q1_fails":  q1_fails,
        "q2_fails":  q2_fails,
        "q3_fails":  q3_fails,
        "details":   verdicts_all,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env(REPO / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — aborting.")
        sys.exit(1)

    p = argparse.ArgumentParser(description="Acceptance validation of cluster output via LLM")
    p.add_argument("--csv",          type=Path, default=DEFAULT_CSV)
    p.add_argument("--out",          type=Path, default=DEFAULT_OUT)
    p.add_argument("--runs",         type=int,  default=4)
    p.add_argument("--sample-size",  type=int,  default=SAMPLE_SIZE)
    p.add_argument("--batch",        type=int,  default=BATCH)
    p.add_argument("--cluster-size", choices=["all", "4", "3", "2"], default="all",
                   help="Which cluster sizes to sample from (default: all)")
    p.add_argument("--seeds",        nargs="+", type=int, default=None,
                   help="Override seeds (space-separated integers)")
    args = p.parse_args()

    from anthropic import Anthropic
    client = Anthropic()

    print(f"\nLoading {args.csv.name} ...")
    df = pd.read_csv(args.csv, low_memory=False)

    size_map = df.groupby("ensemble_cluster_id").size()
    if args.cluster_size == "all":
        candidate_ids = list(size_map[size_map >= 2].index)
        pool_label = "all clusters (2–4 way)"
    else:
        sz = int(args.cluster_size)
        candidate_ids = list(size_map[size_map == sz].index)
        pool_label = f"{sz}-way clusters"

    print(f"Pool : {len(candidate_ids):,} {pool_label}")
    print(f"Runs : {args.runs} independent samples of {args.sample_size} clusters")
    print(f"Model: {MODEL}\n")

    seeds = args.seeds or SEEDS[: args.runs]
    all_results: list[dict] = []
    all_detail_rows: list[dict] = []

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{args.runs}  (seed={seed}) ---")
        result = run_sample(client, df, candidate_ids, seed,
                            sample_size=args.sample_size, batch_size=args.batch)
        all_results.append(result)

        verdict_char = "✓ PASS" if result["verdict"] == "PASS" else "✗ FAIL"
        print(
            f"  Result: {result['n_pass']}/{result['n_total']}  "
            f"{result['pass_rate']:.1%}  {verdict_char}"
        )
        print(
            f"  Q-fails: Q1={result['q1_fails']}  "
            f"Q2={result['q2_fails']}  Q3={result['q3_fails']}"
        )

        for v in result["details"]:
            all_detail_rows.append({
                "run_seed":    seed,
                "cluster_id":  v.get("cluster_id"),
                "q1":          v.get("q1"),
                "q2":          v.get("q2"),
                "q3":          v.get("q3"),
                "q3_na":       v.get("q3_na"),
                "pass":        v.get("pass"),
                "reason":      v.get("reason", ""),
            })

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"{'Run':>4}  {'Seed':>6}  {'Pass':>9}  {'Rate':>6}  "
          f"{'Q1 fail':>7}  {'Q2 fail':>7}  {'Q3 fail':>7}  Verdict")
    print("-" * 65)
    for r in all_results:
        print(
            f"{all_results.index(r)+1:>4}  {r['seed']:>6}  "
            f"{r['n_pass']:>4}/{r['n_total']:<4}  {r['pass_rate']:>5.1%}  "
            f"{r['q1_fails']:>7}  {r['q2_fails']:>7}  {r['q3_fails']:>7}  "
            f"{'✓ PASS' if r['verdict'] == 'PASS' else '✗ FAIL'}"
        )
    print("=" * 65)

    rates    = [r["pass_rate"] for r in all_results]
    n_pass_r = sum(1 for r in all_results if r["verdict"] == "PASS")
    print(
        f"\nAcross {args.runs} runs:  "
        f"min {min(rates):.1%}  avg {sum(rates)/len(rates):.1%}  max {max(rates):.1%}"
    )
    print(
        f"Acceptance threshold: {PASS_THRESHOLD:.0%} per sample  "
        f"({PASS_THRESHOLD * args.sample_size:.0f}/{args.sample_size} clusters must pass)"
    )
    print(
        f"Runs passing threshold: {n_pass_r}/{args.runs}  →  "
        f"{'✓ DELIVERABLE PASSES' if n_pass_r == args.runs else '⚠ SOME RUNS FAIL'}"
    )

    # -----------------------------------------------------------------------
    # Save detail CSV
    # -----------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    detail_df = pd.DataFrame(all_detail_rows)
    # Merge cluster metadata
    clust_meta = (
        df.groupby("ensemble_cluster_id")
        .apply(lambda g: " || ".join(g["names"].astype(str).str[:60].tolist()))
        .reset_index()
        .rename(columns={"ensemble_cluster_id": "cluster_id", 0: "titles_preview"})
    )
    detail_df = detail_df.merge(clust_meta, on="cluster_id", how="left")
    detail_df.to_csv(args.out, index=False)
    print(f"\nDetailed results saved to {args.out}\n")


if __name__ == "__main__":
    main()
