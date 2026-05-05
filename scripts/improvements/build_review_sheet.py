"""
Build the clause 4.2 validation review sheet.

Produces a self-contained HTML form that any unaffiliated reviewer can open in
a browser and fill in. The form asks the three contractual Yes/No questions
per cluster:

    Q1. Are all items in the cluster the exact same core product?
    Q2. Are all items within an acceptable weight variance?
    Q3. For own-brand goods, are they of the same product tier?
         (e.g. all "Value", all "Finest")

When the reviewer clicks Submit, the form serialises every answer to a CSV
file that downloads to their machine. The four CSVs (one per assessor) can
then be aggregated into the contractual 90% pass-rate measure.

Sampling
--------
Stratified random sample of 50 clusters across the cluster-size buckets,
weighted by the deliverable's distribution to mirror what we ship.

Inputs
------
  data/outputs/ensemble/ensemble_clusters.csv

Outputs
-------
  data/outputs/improvements/review_sheet.html      open in browser
  data/outputs/improvements/review_sample.csv      machine-readable sample
  data/outputs/improvements/review_results_blank.csv   schema for aggregation

Usage
-----
    uv run python scripts/improvements/build_review_sheet.py
    uv run python scripts/improvements/build_review_sheet.py --seed 7 --n 50
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLUSTERS = REPO_ROOT / "data/outputs/ensemble/ensemble_clusters.csv"
OUT_HTML = REPO_ROOT / "data/outputs/improvements/review_sheet.html"
OUT_SAMPLE = REPO_ROOT / "data/outputs/improvements/review_sample.csv"
OUT_BLANK = REPO_ROOT / "data/outputs/improvements/review_results_blank.csv"


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = df.groupby("ensemble_cluster_id").size()
    by_size = {s: sizes[sizes == s].index.tolist() for s in (2, 3, 4)}

    counts = {s: len(ids) for s, ids in by_size.items()}
    total = sum(counts.values())
    quota = {s: max(1, round(n * counts[s] / total)) for s in by_size}
    diff = n - sum(quota.values())
    if diff != 0:
        quota[4] += diff

    chosen: list[int] = []
    for s, ids in by_size.items():
        k = min(quota[s], len(ids))
        chosen.extend(rng.choice(ids, size=k, replace=False).tolist())

    return df[df["ensemble_cluster_id"].isin(chosen)].copy()


def render_html(sample: pd.DataFrame) -> str:
    cluster_ids = sorted(sample["ensemble_cluster_id"].unique().tolist())
    blocks: list[str] = []

    for i, cid in enumerate(cluster_ids, start=1):
        items = sample[sample["ensemble_cluster_id"] == cid].copy()
        items = items.sort_values("supermarket")
        rows_html = "\n".join(
            f"""
            <tr>
              <td>{html.escape(str(r['supermarket']))}</td>
              <td>{html.escape(str(r.get('names', '')))}</td>
              <td>{html.escape(str(r.get('unit_value', '')))} {html.escape(str(r.get('unit_type', '')))}</td>
              <td>£{html.escape(str(r.get('prices_(£)', '')))}</td>
              <td>{html.escape(str(r.get('tier_keyword', '') or '—'))}</td>
              <td>{'own-brand' if str(r.get('own_brand', '')).lower() == 'true' else 'branded'}</td>
            </tr>
            """
            for _, r in items.iterrows()
        )

        blocks.append(f"""
        <section class="cluster" data-cluster-id="{cid}">
          <header>
            <h3>Cluster {i} of {len(cluster_ids)} <span class="cid">(id #{cid})</span></h3>
          </header>
          <table>
            <thead>
              <tr><th>Supermarket</th><th>Product name</th><th>Pack size</th><th>Price</th><th>Tier</th><th>Type</th></tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
          <div class="questions">
            <fieldset>
              <legend>Q1. Are all items the <strong>exact same core product</strong>?</legend>
              <label><input type="radio" name="q1_{cid}" value="yes" required> Yes</label>
              <label><input type="radio" name="q1_{cid}" value="no"> No</label>
            </fieldset>
            <fieldset>
              <legend>Q2. Are all items within an <strong>acceptable weight variance</strong>?</legend>
              <label><input type="radio" name="q2_{cid}" value="yes" required> Yes</label>
              <label><input type="radio" name="q2_{cid}" value="no"> No</label>
            </fieldset>
            <fieldset>
              <legend>Q3. For own-brand goods, are they of the <strong>same product tier</strong>? <span class="hint">(answer "yes" if not applicable — i.e. no own-brand items in this cluster)</span></legend>
              <label><input type="radio" name="q3_{cid}" value="yes" required> Yes</label>
              <label><input type="radio" name="q3_{cid}" value="no"> No</label>
              <label><input type="radio" name="q3_{cid}" value="na"> N/A</label>
            </fieldset>
            <fieldset class="notes">
              <legend>Optional notes</legend>
              <input type="text" name="note_{cid}" placeholder="(optional)" maxlength="200">
            </fieldset>
          </div>
        </section>
        """)

    cluster_ids_json = json.dumps(cluster_ids)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ShopWiser cluster review (clause 4.2)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 960px; margin: 2rem auto; padding: 0 1.25rem; color: #1d2433; }}
  h1 {{ font-size: 1.6rem; }}
  h3 {{ margin: 0; font-size: 1.05rem; }}
  .cid {{ color: #888; font-weight: normal; font-size: 0.85em; }}
  .meta {{ background: #f4f6fb; padding: 0.85rem 1rem; border-radius: 8px;
           border: 1px solid #dfe4ee; margin-bottom: 1.5rem; font-size: 0.93rem; }}
  .meta input[type=text] {{ padding: 0.35rem 0.5rem; border: 1px solid #c2cad8;
                             border-radius: 6px; width: 18rem; }}
  section.cluster {{ border: 1px solid #dfe4ee; border-radius: 10px;
                     padding: 1rem 1.1rem; margin: 1rem 0; background: #fff; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.6rem 0 1rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.55rem; font-size: 0.92rem;
            border-bottom: 1px solid #ecf0f6; }}
  th {{ background: #f8fafc; font-weight: 600; }}
  fieldset {{ border: none; margin: 0.4rem 0; padding: 0; }}
  legend {{ font-weight: 500; padding-bottom: 0.25rem; }}
  .hint {{ color: #6a7488; font-weight: 400; font-size: 0.85em; }}
  label {{ margin-right: 1.4rem; cursor: pointer; }}
  .notes input {{ width: 100%; padding: 0.4rem 0.55rem; border: 1px solid #c2cad8;
                  border-radius: 6px; }}
  button {{ background: #2E86AB; color: white; border: 0; padding: 0.7rem 1.4rem;
            border-radius: 8px; font-size: 1rem; cursor: pointer; }}
  button:hover {{ background: #226d8e; }}
  .footer {{ position: sticky; bottom: 0; padding: 1rem 0;
             background: rgba(255,255,255,0.96); border-top: 1px solid #dfe4ee;
             text-align: right; }}
</style>
</head>
<body>
  <h1>ShopWiser cluster review</h1>
  <p>You are reviewing {len(cluster_ids)} randomly-sampled clusters from the matching pipeline. For each cluster please answer the three Yes/No questions below. Your answers will download as a CSV when you click <em>Submit</em> at the bottom — please send that file back.</p>

  <div class="meta">
    <label>Reviewer name:&nbsp;<input type="text" id="reviewer_name" placeholder="(your name)"></label>
  </div>

  <form id="review-form">
    {''.join(blocks)}
    <div class="footer">
      <button type="submit">Submit and download answers</button>
    </div>
  </form>

<script>
const CLUSTER_IDS = {cluster_ids_json};

document.getElementById('review-form').addEventListener('submit', function(ev) {{
  ev.preventDefault();
  const reviewer = document.getElementById('reviewer_name').value.trim() || 'unnamed';
  const rows = [['reviewer', 'cluster_id', 'q1_same_product', 'q2_weight_variance', 'q3_same_tier', 'cluster_passed', 'notes']];
  let missing = [];
  CLUSTER_IDS.forEach(function(cid) {{
    const q1 = (document.querySelector('input[name="q1_'+cid+'"]:checked') || {{}}).value;
    const q2 = (document.querySelector('input[name="q2_'+cid+'"]:checked') || {{}}).value;
    const q3 = (document.querySelector('input[name="q3_'+cid+'"]:checked') || {{}}).value;
    const note = (document.querySelector('input[name="note_'+cid+'"]') || {{}}).value || '';
    if (!q1 || !q2 || !q3) {{ missing.push(cid); return; }}
    const passed = (q1 === 'yes' && q2 === 'yes' && (q3 === 'yes' || q3 === 'na')) ? 'yes' : 'no';
    rows.push([reviewer, cid, q1, q2, q3, passed, note.replace(/[",\\r\\n]+/g, ' ')]);
  }});
  if (missing.length) {{
    alert('Please answer every question. Missing answers for clusters: ' + missing.join(', '));
    return;
  }}
  const csv = rows.map(r => r.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(',')).join('\\n');
  const blob = new Blob([csv], {{type: 'text/csv;charset=utf-8;'}});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'shopwiser_review_' + reviewer.replace(/[^a-z0-9]+/gi, '_') + '.csv';
  document.body.appendChild(link); link.click(); document.body.removeChild(link);
  alert('Thanks! Your answers have been downloaded as a CSV. Please send the file back.');
}});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading clusters: {CLUSTERS}")
    df = pd.read_csv(CLUSTERS, low_memory=False)
    print(f"  {len(df):,} rows / {df['ensemble_cluster_id'].nunique():,} clusters\n")

    sample = stratified_sample(df, n=args.n, seed=args.seed)
    n_chosen = sample["ensemble_cluster_id"].nunique()
    by_size = sample.groupby("ensemble_cluster_id").size().value_counts().sort_index()
    print(f"Sampled {n_chosen} clusters (seed={args.seed}):")
    for sz, cnt in by_size.items():
        print(f"  {sz}-way : {cnt}")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(render_html(sample), encoding="utf-8")
    print(f"\nWrote review form  : {OUT_HTML}")

    sample_cols = [
        "ensemble_cluster_id", "supermarket", "names", "prices_(£)",
        "unit_value", "unit_type", "tier_keyword", "own_brand", "category",
    ]
    keep = [c for c in sample_cols if c in sample.columns]
    sample[keep].sort_values(["ensemble_cluster_id", "supermarket"]).to_csv(OUT_SAMPLE, index=False)
    print(f"Wrote sample CSV   : {OUT_SAMPLE}")

    blank = pd.DataFrame(columns=[
        "reviewer", "cluster_id", "q1_same_product", "q2_weight_variance",
        "q3_same_tier", "cluster_passed", "notes",
    ])
    blank.to_csv(OUT_BLANK, index=False)
    print(f"Wrote blank schema : {OUT_BLANK}")
    print("\nNext step: open the HTML in a browser, hand to 4 unaffiliated reviewers,")
    print("collect 4 CSVs, then aggregate (cluster passes when ≥3 of 4 reviewers said pass).")


if __name__ == "__main__":
    main()
