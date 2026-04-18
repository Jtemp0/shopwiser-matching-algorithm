"""Export a hand-validated cofounder demo slice from ``ml_clusters.csv``.

The **long** CSV keeps every pipeline column (normalisation, units, attributes, etc.).
The **wide** CSV is one row per cluster with retailer product title, size, and price.
"""

from __future__ import annotations

import argparse
import html
import random
from pathlib import Path

import pandas as pd

from shopwiser.paths import DATA_OUTPUTS, PROJECT_ROOT

DEMO_DIR = DATA_OUTPUTS / 'demo'

# Twenty-five exact 4-way clusters (one row per supermarket), manually reviewed.
# Selection rules: identical ``normalized_name`` across all four rows; unit_value
# either identical or within ~3% relative spread (where numeric); same branded SKU
# or clear own-label equivalent (e.g. chopped ginger). Replaces auto-sampled clusters
# that mixed different products (e.g. cake blob, spread vs pudding).
VALIDATED_COFOUNDER_DEMO_CLUSTER_IDS: tuple[int, ...] = (
    2728,  # Lavazza Qualità Rossa coffee beans 1kg
    2813,  # Ginsters Peppered Steak Slice
    2854,  # Peperami Firestick
    2881,  # Nature's Finest Peach in Juice
    2892,  # Urban Fruit Gently Baked Mango
    2945,  # Schweppes Pink Soda 1L
    3078,  # Robinsons Pressed Pear & Elderflower cordial
    3113,  # Batchelors Cup a Soup Chicken Noodle
    3239,  # McGuigan Black Label Pinot Grigio
    3271,  # McCain Potato Smiles (450/454g regional pack)
    3282,  # Goodfella's GF Pepperoni Pizza (317/320g)
    3337,  # Own-label chopped ginger 75g
    3386,  # Onken Cherry Yogurt 450g
    3552,  # Heinz Macaroni Cheese 400g
    3661,  # Lea & Perrins Worcestershire Sauce 150ml
    3677,  # Filippo Berio Mild & Light Olive Oil 500ml
    3683,  # Yutaka Japanese Rice Vinegar 150ml
    3815,  # Blue Dragon Hoisin & Garlic stir-fry sauce 120g
    3869,  # Warburtons Gluten Free Multiseed Loaf 300g
    3877,  # Real Lancashire Eccles Cakes (same line; unit not in scrape)
    3948,  # Own-label light soy sauce 150ml
    4011,  # Pukka All Day Breakfast Slice
    4097,  # Bonne Maman Peach Conserve (370/375g pack drift)
    4211,  # Rustlers All Day Breakfast Pancake Stack
    4239,  # Quorn Mini Sausage Rolls
)

# Stable column order for the wide CSV (UK retailers).
SM_ORDER = ('Tesco', 'Sainsbury\'s', 'ASDA', 'Morrisons')
SM_FROM_CSV = {
    'Tesco': 'Tesco',
    'Sains': 'Sainsbury\'s',
    'ASDA': 'ASDA',
    'Morrisons': 'Morrisons',
}


def _size_label(row: pd.Series) -> str:
    uv = row.get('unit_value')
    ut = row.get('unit_type')
    if pd.isna(uv) or not ut or (isinstance(ut, float) and pd.isna(ut)):
        return ''
    try:
        v = float(uv)
        if v == int(v):
            v = int(v)
        return f'{v}{ut}'
    except (TypeError, ValueError):
        return ''


def _exact_fourway_cluster_ids(df: pd.DataFrame) -> list[int]:
    multi = df[df['cluster_size'] >= 2]
    out: list[int] = []
    for cid, g in multi.groupby('cluster_id'):
        if len(g) == 4 and g['supermarket'].nunique() == 4:
            out.append(int(cid))
    return out


def _pick_demo_clusters_stratified(
    df: pd.DataFrame,
    *,
    n_total: int,
    seed: int,
) -> list[int]:
    """Stratified sample across categories (exact 4-way only). Fallback only."""
    ids = _exact_fourway_cluster_ids(df)
    sub = df[df['cluster_id'].isin(ids)].drop_duplicates('cluster_id')
    cat_map = sub.set_index('cluster_id')['category'].to_dict()
    by_cat: dict[str, list[int]] = {}
    for cid, cat in cat_map.items():
        by_cat.setdefault(str(cat), []).append(int(cid))

    rng = random.Random(seed)
    caps = {
        'food_cupboard': 5,
        'fresh_food': 4,
        'drinks': 4,
        'free-from': 2,
        'frozen': 2,
        'bakery': 1,
    }
    picked: list[int] = []
    remaining = set(ids)

    for cat, cap in caps.items():
        if cat not in by_cat:
            continue
        cids = [c for c in by_cat[cat] if c in remaining]
        rng.shuffle(cids)
        take = min(cap, len(cids), max(0, n_total - len(picked)))
        for p in cids[:take]:
            picked.append(p)
            remaining.discard(p)
        if len(picked) >= n_total:
            break

    if len(picked) < n_total and remaining:
        extra = list(remaining)
        rng.shuffle(extra)
        need = n_total - len(picked)
        picked.extend(extra[:need])

    return sorted(picked[:n_total])


def _write_html(wide: pd.DataFrame, path: Path, *, n_clusters: int) -> None:
    parts: list[str] = [
        '<!DOCTYPE html>',
        '<html lang="en"><head><meta charset="utf-8">',
        '<title>ShopWiser — ML matching demo sample</title>',
        '<style>',
        'body{font-family:system-ui,Segoe UI,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;'
        'color:#1a1a1a;line-height:1.45;}',
        'h1{font-size:1.35rem;font-weight:650;margin-bottom:0.25rem;}',
        'p.lead{color:#444;margin:0 0 1rem 0;}',
        'table{border-collapse:collapse;width:100%;margin:1.25rem 0;font-size:0.88rem;}',
        'th,td{border:1px solid #ccc;padding:8px 10px;vertical-align:top;}',
        'th{background:#f4f4f4;text-align:left;}',
        'tr:nth-child(even){background:#fafafa;}',
        '.meta{color:#666;font-size:0.9rem;margin-bottom:1.5rem;}',
        '.cluster{margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid #e0e0e0;}',
        '.cluster h2{font-size:1rem;margin:0 0 0.5rem 0;}',
        '.tag{display:inline-block;background:#e8f4ea;color:#1b5e20;padding:2px 8px;'
        'border-radius:4px;font-size:0.75rem;margin-right:6px;}',
        '</style></head><body>',
        '<h1>ShopWiser — ML matching demo sample</h1>',
        '<p class="lead">Illustrative <strong>4-way</strong> clusters (one comparable SKU per retailer) '
        'from the conservative semantic + keyword matching pipeline. '
        'Precision is prioritised over recall.</p>',
        f'<p class="meta">{n_clusters} clusters · Prices in £ · Open this file in any browser.</p>',
    ]

    for _, r in wide.iterrows():
        cid = int(r['cluster_id'])
        cat = html.escape(str(r['category']))
        label = html.escape(str(r['cluster_label']))
        parts.append('<section class="cluster">')
        parts.append(f'<h2><span class="tag">{cat}</span> {label}</h2>')
        parts.append('<table><thead><tr>')
        parts.append('<th>Retailer</th><th>Product</th><th>Size</th><th>Price (£)</th></tr></thead><tbody>')
        for sm in SM_ORDER:
            prod = r.get(f'{sm}_product', '')
            price = r.get(f'{sm}_price_£', '')
            size = r.get(f'{sm}_size', '')
            parts.append(
                '<tr>'
                f'<td><strong>{html.escape(sm)}</strong></td>'
                f'<td>{html.escape(str(prod))}</td>'
                f'<td>{html.escape(str(size))}</td>'
                f'<td>{html.escape(str(price))}</td>'
                '</tr>'
            )
        parts.append('</tbody></table></section>')

    parts.append('</body></html>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def export_cofounder_demo(
    ml_clusters_csv: Path | None = None,
    out_dir: Path | None = None,
    *,
    cluster_ids: tuple[int, ...] | None = None,
    use_stratified_fallback: bool = False,
    n_clusters: int = 25,
    seed: int = 42,
    write_html: bool = False,
) -> tuple[Path, Path, Path | None]:
    """Write long CSV, wide CSV, and optionally HTML under ``data/outputs/demo/``.

    By default uses ``VALIDATED_COFOUNDER_DEMO_CLUSTER_IDS``. Set
    ``use_stratified_fallback=True`` to ignore that list and sample automatically.

    Returns ``(long_csv, wide_csv, html_or_none)``.
    """
    csv_in = ml_clusters_csv or (PROJECT_ROOT / 'data/outputs/ml_clusters/ml_clusters.csv')
    out = out_dir or DEMO_DIR
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_in)

    if use_stratified_fallback:
        demo_ids = _pick_demo_clusters_stratified(df, n_total=n_clusters, seed=seed)
    elif cluster_ids is not None:
        demo_ids = list(cluster_ids)
    else:
        demo_ids = list(VALIDATED_COFOUNDER_DEMO_CLUSTER_IDS)
        if n_clusters != len(demo_ids):
            demo_ids = demo_ids[:n_clusters]

    missing = [c for c in demo_ids if c not in set(df['cluster_id'])]
    if missing:
        raise ValueError(f'cluster_id not in ml_clusters.csv: {missing}')

    order_map = {cid: i + 1 for i, cid in enumerate(demo_ids)}
    demo = df[df['cluster_id'].isin(demo_ids)].copy()
    demo['demo_order'] = demo['cluster_id'].map(order_map)
    demo.sort_values(['demo_order', 'supermarket'], inplace=True)

    long_path = out / 'cofounder_demo_clusters_long.csv'
    demo.to_csv(long_path, index=False)

    rows: list[dict] = []
    price_col = 'prices_(£)'
    for cid in sorted(demo_ids, key=lambda x: order_map.get(x, 0)):
        g = demo[demo['cluster_id'] == cid]
        label = g['core_product_name'].dropna().astype(str).iloc[0] if len(g) else ''
        if not label:
            label = g['normalized_name'].iloc[0]
        cat = g['category'].iloc[0]
        row: dict = {
            'demo_order': order_map[cid],
            'cluster_id': cid,
            'category': cat,
            'cluster_label': label,
        }
        for sm_csv, sm_disp in SM_FROM_CSV.items():
            sg = g[g['supermarket'] == sm_csv]
            if sg.empty:
                row[f'{sm_disp}_product'] = ''
                row[f'{sm_disp}_price_£'] = ''
                row[f'{sm_disp}_size'] = ''
                continue
            best = sg.loc[sg[price_col].astype(float).idxmin()]
            row[f'{sm_disp}_product'] = best.get('original_name', best.get('names', ''))
            row[f'{sm_disp}_price_£'] = round(float(best[price_col]), 2)
            row[f'{sm_disp}_size'] = _size_label(best)
        rows.append(row)

    wide = pd.DataFrame(rows)
    wide_path = out / 'cofounder_demo_clusters_wide.csv'
    wide.to_csv(wide_path, index=False)

    html_path: Path | None = None
    if write_html:
        html_path = out / 'cofounder_demo.html'
        _write_html(wide, html_path, n_clusters=len(demo_ids))

    return long_path, wide_path, html_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description='Export cofounder demo tables from ml_clusters.csv')
    p.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Path to ml_clusters.csv (default: data/outputs/ml_clusters/ml_clusters.csv)',
    )
    p.add_argument('--out-dir', type=Path, default=None, help='Output directory (default: data/outputs/demo)')
    p.add_argument('--n-clusters', type=int, default=25, help='When using --stratified, number of clusters')
    p.add_argument('--seed', type=int, default=42, help='Random seed for stratified sampling')
    p.add_argument(
        '--stratified',
        action='store_true',
        help='Sample clusters automatically instead of the validated ID list',
    )
    p.add_argument('--html', action='store_true', help='Also write cofounder_demo.html')
    args = p.parse_args(argv)

    long_p, wide_p, html_p = export_cofounder_demo(
        ml_clusters_csv=args.input,
        out_dir=args.out_dir,
        use_stratified_fallback=args.stratified,
        n_clusters=args.n_clusters,
        seed=args.seed,
        write_html=args.html,
    )
    print('Wrote:')
    print(f'  {long_p}')
    print(f'  {wide_p}')
    if html_p:
        print(f'  {html_p}')


if __name__ == '__main__':
    main()
