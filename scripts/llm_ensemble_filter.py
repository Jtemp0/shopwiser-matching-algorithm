"""
LLM-based ensemble filter — post-processing pass for the relaxed Kruskal ensemble.

The relaxed ensemble (Jaccard floor lowered to 0.15) recovers coverage but admits
borderline clusters that need semantic verification.  This script classifies every
cluster into one of three tiers and verifies the borderline tier with Claude Haiku
using the same Q1/Q2/Q3 acceptance questions as the validation step.

Tiers
-----
  AUTO-PASS   min-pair Jaccard ≥ 0.65  →  include directly (high-confidence names)
  LLM-VERIFY  0.15 ≤ min-pair Jaccard < 0.65  →  Haiku Q1/Q2/Q3; include only if pass
  AUTO-REJECT min-pair Jaccard < 0.15  →  drop (already blocked by ensemble gate)

Usage
-----
  uv run python scripts/llm_ensemble_filter.py
  uv run python scripts/llm_ensemble_filter.py --dry-run   # classify only, no API calls
  uv run python scripts/llm_ensemble_filter.py --jaccard-threshold 0.60
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL              = "claude-haiku-4-5-20251001"
DEFAULT_CSV        = REPO / "data/intermediate/ensemble_clusters.csv"
DEFAULT_OUT        = REPO / "data/intermediate/ensemble_clusters.csv"  # overwrite in-place
REPORT_OUT         = REPO / "data/intermediate/llm_filter_report.csv"
BATCH              = 10    # clusters per API call
AUTO_PASS_JACCARD  = 0.65  # min-pair Jaccard ≥ this → skip LLM
SLEEP_BETWEEN      = 0.4   # seconds between batches


# ---------------------------------------------------------------------------
# Env helper
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
# Jaccard helper
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    ta = set(str(a).lower().split())
    tb = set(str(b).lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _min_pair_jaccard(names: list[str]) -> float:
    """Minimum pairwise Jaccard across all name pairs in the cluster."""
    if len(names) < 2:
        return 1.0
    mn = 1.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mn = min(mn, _jaccard(names[i], names[j]))
    return mn


# ---------------------------------------------------------------------------
# Cluster display helpers (identical to contract_validate.py)
# ---------------------------------------------------------------------------

def _tier_label(row: pd.Series) -> str:
    t = str(row.get("tier_type", "")).lower()
    if t == "premium":   return "premium-tier"
    if t == "value":     return "value-tier"
    if t == "standard":  return "standard-tier"
    if str(row.get("product_type", "")).lower() == "branded":
        return "branded"
    return "unknown-tier"


def _weight_variance_summary(g: pd.DataFrame) -> str:
    sizes = []
    for _, r in g.iterrows():
        uv = r.get("unit_value")
        if pd.notna(uv) and uv:
            try:
                v = float(uv)
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
    verdict = "within 15% → Q2 = Yes" if within else "EXCEEDS 15% → Q2 = No"
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
        tier     = _tier_label(r)
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
# Prompt (identical to contract_validate.py)
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


# ---------------------------------------------------------------------------
# Haiku call
# ---------------------------------------------------------------------------

def _call_llm(client, cluster_ids: list[int], df: pd.DataFrame) -> list[dict]:
    blocks = "\n\n".join(_cluster_block(df, cid) for cid in cluster_ids)
    user = USER_TEMPLATE.format(ids=cluster_ids, blocks=blocks)
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
# Classification
# ---------------------------------------------------------------------------

def classify_clusters(
    df: pd.DataFrame,
    auto_pass_threshold: float = AUTO_PASS_JACCARD,
) -> tuple[list[int], list[int]]:
    """Return (auto_pass_ids, llm_verify_ids).

    auto_pass   — min-pair Jaccard ≥ threshold (high-confidence names)
    llm_verify  — min-pair Jaccard < threshold but ≥ 0.15 (borderline)
    Clusters with min-pair Jaccard < 0.15 are already blocked by the
    ensemble gate and should never appear here.
    """
    auto_pass: list[int] = []
    llm_verify: list[int] = []

    for cid, grp in df.groupby("ensemble_cluster_id"):
        names = grp["normalized_name"].dropna().tolist()
        min_j = _min_pair_jaccard(names)
        if min_j >= auto_pass_threshold:
            auto_pass.append(int(cid))
        else:
            llm_verify.append(int(cid))

    return auto_pass, llm_verify


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LLM-based ensemble filter")
    parser.add_argument("--input",  default=str(DEFAULT_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUT),
                        help="Destination CSV (default: overwrite input in-place)")
    parser.add_argument("--report", default=str(REPORT_OUT))
    parser.add_argument("--jaccard-threshold", type=float, default=AUTO_PASS_JACCARD,
                        help="Min-pair Jaccard ≥ this → auto-pass without LLM (default 0.65)")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify only; skip API calls and do not rewrite CSV")
    args = parser.parse_args(argv)

    _load_env(REPO / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("ANTHROPIC_API_KEY not set.")

    print(f"Loading ensemble: {args.input}")
    df = pd.read_csv(args.input)
    total_clusters = df["ensemble_cluster_id"].nunique()
    print(f"  {total_clusters:,} clusters, {len(df):,} rows")

    # --- classify ---
    auto_pass_ids, llm_verify_ids = classify_clusters(df, args.jaccard_threshold)
    print(f"\nClassification (Jaccard threshold = {args.jaccard_threshold}):")
    print(f"  AUTO-PASS  (min Jaccard ≥ {args.jaccard_threshold}): {len(auto_pass_ids):,} clusters")
    print(f"  LLM-VERIFY (min Jaccard <  {args.jaccard_threshold}): {len(llm_verify_ids):,} clusters")

    if args.dry_run:
        print("\n--dry-run: skipping API calls.")
        return

    # --- LLM verification ---
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    approved_ids: list[int] = []
    rejected_ids: list[int] = []
    report_rows:  list[dict] = []

    total_batches = (len(llm_verify_ids) + args.batch - 1) // args.batch
    print(f"\nVerifying {len(llm_verify_ids):,} borderline clusters in "
          f"{total_batches} batches of ≤{args.batch}...")

    for b_idx in range(0, len(llm_verify_ids), args.batch):
        batch = llm_verify_ids[b_idx : b_idx + args.batch]
        batch_num = b_idx // args.batch + 1
        try:
            verdicts = _call_llm(client, batch, df)
        except Exception as exc:
            print(f"  batch {batch_num}/{total_batches} ERROR: {exc} — marking all as REJECTED")
            for cid in batch:
                rejected_ids.append(cid)
                report_rows.append({"cluster_id": cid, "tier": "llm_verify",
                                    "llm_pass": False, "reason": f"API error: {exc}",
                                    "q1": "?", "q2": "?", "q3": "?"})
            continue

        verdict_map = {v["cluster_id"]: v for v in verdicts}
        for cid in batch:
            v = verdict_map.get(cid)
            if v is None:
                rejected_ids.append(cid)
                report_rows.append({"cluster_id": cid, "tier": "llm_verify",
                                    "llm_pass": False, "reason": "no verdict returned",
                                    "q1": "?", "q2": "?", "q3": "?"})
                continue
            passed = bool(v.get("pass", False))
            if passed:
                approved_ids.append(cid)
            else:
                rejected_ids.append(cid)
            report_rows.append({
                "cluster_id": cid,
                "tier": "llm_verify",
                "llm_pass": passed,
                "reason": v.get("reason", ""),
                "q1": v.get("q1", "?"),
                "q2": v.get("q2", "?"),
                "q3": v.get("q3", "?"),
            })

        done = min(b_idx + args.batch, len(llm_verify_ids))
        pct  = done / len(llm_verify_ids) * 100
        n_approved = len(approved_ids)
        n_rejected = len(rejected_ids)
        print(f"  [{pct:5.1f}%]  batch {batch_num}/{total_batches}  "
              f"approved so far: {n_approved}  rejected so far: {n_rejected}")
        time.sleep(SLEEP_BETWEEN)

    # Add auto-pass entries to report (no LLM call)
    for cid in auto_pass_ids:
        report_rows.append({"cluster_id": cid, "tier": "auto_pass",
                             "llm_pass": True, "reason": "Jaccard ≥ threshold",
                             "q1": "n/a", "q2": "n/a", "q3": "n/a"})

    # --- write report ---
    report_df = pd.DataFrame(report_rows)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(args.report, index=False)

    # --- filter ensemble CSV ---
    keep_ids = set(auto_pass_ids) | set(approved_ids)
    df_filtered = df[df["ensemble_cluster_id"].isin(keep_ids)].copy()

    df_filtered.to_csv(args.output, index=False)

    # --- summary ---
    total_kept    = df_filtered["ensemble_cluster_id"].nunique()
    total_removed = total_clusters - total_kept
    llm_approved  = len(approved_ids)
    llm_rejected  = len(rejected_ids)
    llm_approval_rate = llm_approved / max(1, len(llm_verify_ids)) * 100

    print(f"\n{'='*60}")
    print(f"Filter summary:")
    print(f"  Auto-passed (no LLM):   {len(auto_pass_ids):>6,}")
    print(f"  LLM-verified total:     {len(llm_verify_ids):>6,}")
    print(f"    → approved:           {llm_approved:>6,}  ({llm_approval_rate:.1f}%)")
    print(f"    → rejected:           {llm_rejected:>6,}")
    print(f"  Total clusters kept:    {total_kept:>6,}  (removed {total_removed:,})")
    print(f"")

    # Breakdown by cluster size
    size_dist = df_filtered.groupby("cluster_size")["ensemble_cluster_id"].nunique()
    for sz in sorted(size_dist.index):
        label = {2: "2-way", 3: "3-way", 4: "4-way"}.get(sz, f"{sz}-way")
        print(f"  {label}: {size_dist[sz]:,}")

    print(f"\nFiltered ensemble written to: {args.output}")
    print(f"Report written to:            {args.report}")


if __name__ == "__main__":
    main()
