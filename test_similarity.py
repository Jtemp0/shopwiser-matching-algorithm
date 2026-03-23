"""
ShopWiser — Similarity Self-Tests
==================================

All test fixtures for compute_similarity live here and nowhere else.
Run standalone:
    python test_similarity.py

Or call run_tests() from any script:
    from test_similarity import run_tests
    all_pass = run_tests(my_compute_similarity, tol_branded=0.05,
                         tol_own=0.05, tol_unbranded=0.05)

NOTE: When constants change in clustering_v5.py (SHORT_STRIPPED_THRESHOLD,
UNIT_TOLERANCE_*, FUZZY_THRESHOLD_*), re-verify the expected outcomes here.
"""

import sys
import pandas as pd


def _mk(name, uv, ut, sm, brand=None, tier='standard', ptype='branded',
        pq=None, attrs=None, trunc=False):
    """Build a minimal pandas.Series row for use in similarity tests."""
    return pd.Series({
        'normalized_name':   name,
        'unit_value':        uv,
        'unit_type':         ut,
        'supermarket':       sm,
        'has_unit':          pd.notna(uv) and uv > 0,
        'pack_quantity':     pq,
        'attributes_keywords': attrs,
        'known_brand_clean': brand,
        'tier_type':         tier,
        'product_type':      ptype,
        'is_truncated':      trunc,
    })


def build_test_cases(tol_branded, tol_own, tol_unbranded):
    """Return list of (description, row_a, row_b, pass_type, tolerance, expected_match)."""
    return [

        # ── original v4 tests ────────────────────────────────────────────────
        ('Same product diff SM',
         _mk('baked beans tomato sauce', 415, 'g', 'Tesco', brand='heinz'),
         _mk('baked beans tomato sauce', 415, 'g', 'ASDA',  brand='heinz'),
         'branded', tol_branded, True),

        ('Plum vs Chopped tomatoes',
         _mk('plum tomatoes', 400, 'g', 'Tesco'),
         _mk('chopped tomatoes', 400, 'g', 'ASDA'),
         'unbranded', tol_unbranded, False),

        ('Same supermarket',
         _mk('whole milk', 2270, 'g', 'Tesco'),
         _mk('whole milk', 2270, 'g', 'Tesco'),
         'unbranded', tol_unbranded, False),

        ('Weight mismatch 200g vs 400g',
         _mk('sweetcorn water', 200, 'g', 'Tesco'),
         _mk('sweetcorn water', 400, 'g', 'ASDA'),
         'unbranded', tol_unbranded, False),

        ('Unit type mismatch g vs ml',
         _mk('orange juice', 1000, 'g', 'Tesco'),
         _mk('orange juice', 1000, 'ml', 'ASDA'),
         'unbranded', tol_unbranded, False),

        ('Chocolate slices vs Cherry bakewell → must NOT match',
         _mk('chocolate slices', None, None, 'Tesco', brand='mr kipling'),
         _mk('cherry bakewell cakes', None, None, 'ASDA', brand='mr kipling'),
         'branded', tol_branded, False),

        ('Rhubarb kefir vs Honey kefir → must NOT match',
         _mk('kefir rhubarb fermented organic yogurt', 350, 'g', 'Tesco', brand='yeo valley'),
         _mk('organic kefir honey yogurt', 350, 'g', 'ASDA', brand='yeo valley'),
         'branded', tol_branded, False),

        ('19 Crimes Rosé vs Red wine → must NOT match',
         _mk('revolutionary rose', 750, 'ml', 'Tesco', brand='19 crimes'),
         _mk('red wine', 750, 'ml', 'ASDA', brand='19 crimes'),
         'branded', tol_branded, False),

        ('Tuna spring water vs olive oil → must NOT match',
         _mk('tuna chunks in spring water', 145, 'g', 'Tesco', brand=None, ptype='own_brand'),
         _mk('tuna chunks in olive oil',    145, 'g', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        ('Highland Park 12yr vs 10yr → must NOT match',
         _mk('highland park 12 year old single malt scotch whisky', 700, 'ml', 'Tesco', brand=None, ptype='unbranded'),
         _mk('highland park 10 year old single malt scotch whisky', 700, 'ml', 'ASDA',  brand=None, ptype='unbranded'),
         'unbranded', tol_unbranded, False),

        ('Pack mismatch 4x vs 10x → must NOT match',
         _mk('alcohol free draught stout', 440, 'ml', 'Tesco', brand='guinness', pq=4),
         _mk('alcohol free draught stout', 440, 'ml', 'ASDA',  brand='guinness', pq=10),
         'branded', tol_branded, False),

        ('Honey Cheerios 515g vs Multigrain Cheerios 540g → must NOT match',
         _mk('honey cheerios cereal',     515, 'g', 'Tesco', brand='cheerios'),
         _mk('cheerios multigrain cereal', 540, 'g', 'ASDA',  brand='cheerios'),
         'branded', tol_branded, False),

        # Meat/ingredient variety conflict — was blocked by SHORT_STRIPPED_THRESHOLD (0.92);
        # now caught earlier by FLAVOR_NAMED_TOKENS ingredient conflict (lamb vs ham).
        ('Knorr lamb stock cubes vs ham stock cubes → must NOT match',
         _mk('stock cubes lamb', 104, 'g', 'Tesco', brand='knorr'),
         _mk('stock cubes ham',  104, 'g', 'ASDA',  brand='knorr'),
         'branded', tol_branded, False),

        ('Schwartz garam masala vs garam masala jar → SHOULD match',
         _mk('garam masala',     29.9, 'g', 'Tesco', brand='schwartz'),
         _mk('garam masala jar', 29.9, 'g', 'ASDA',  brand='schwartz'),
         'branded', tol_branded, True),

        ('Black beans vs Black eyed beans (own-brand) → must NOT match',
         _mk('black beans',      400, 'g', 'Tesco', brand=None, ptype='own_brand'),
         _mk('black eyed beans', 400, 'g', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        ('Malbec Rosé vs Malbec red wine → must NOT match',
         _mk('malbec rose wine', 750, 'ml', 'Tesco', brand=None, ptype='unbranded'),
         _mk('malbec red wine',  750, 'ml', 'ASDA',  brand=None, ptype='unbranded'),
         'unbranded', tol_unbranded, False),

        ('No Added Sugar squash vs regular squash → must NOT match',
         _mk('blackcurrant squash no added sugar', 1000, 'ml', 'Tesco', brand=None, ptype='own_brand'),
         _mk('blackcurrant squash',                1000, 'ml', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        ('Skimmed vs Semi-skimmed milk → must NOT match',
         _mk('skimmed milk',      2000, 'ml', 'Tesco', brand=None, ptype='own_brand'),
         _mk('semi skimmed milk', 2000, 'ml', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        # ── v5 new tests ──────────────────────────────────────────────────────
        ('v5: Alpro Soya vs Alpro Oat → must NOT match (milk base conflict)',
         _mk('soya drink', 1000, 'ml', 'Tesco', brand='alpro'),
         _mk('oat drink',  1000, 'ml', 'ASDA',  brand='alpro'),
         'branded', tol_branded, False),

        ("v5: Green & Black's Ginger vs Mint → must NOT match (flavor conflict)",
         _mk('ginger dark chocolate bar', 100, 'g', 'Tesco', brand="green & black's"),
         _mk('mint dark chocolate bar',   100, 'g', 'ASDA',  brand="green & black's"),
         'branded', tol_branded, False),

        ('v5: Old Mout Cider vs Old Mout Alcohol Free → must NOT match (alcohol-free)',
         _mk('old mout cider berries cherries',        500, 'ml', 'Tesco', brand='old mout'),
         _mk('old mout alcohol free berries cherries', 500, 'ml', 'ASDA',  brand='old mout'),
         'branded', tol_branded, False),

        ('v5: Raw chicken breast vs Roasted chicken breast → must NOT match (cooking state)',
         _mk('chicken breast raw',     600, 'g', 'Tesco', brand=None, ptype='own_brand'),
         _mk('chicken breast roasted', 600, 'g', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        ('v5: Baby carrots vs Carrots → must NOT match (one-sided baby token)',
         _mk('baby carrots', 300, 'g', 'Tesco', brand=None, ptype='unbranded'),
         _mk('carrots',      300, 'g', 'ASDA',  brand=None, ptype='unbranded'),
         'unbranded', tol_unbranded, False),

        ('v5: Single vs 6-pack (NaN pack vs multipack) → must NOT match',
         _mk('still water', 500, 'ml', 'Tesco', brand='evian', pq=None),
         _mk('still water', 500, 'ml', 'ASDA',  brand='evian', pq=6),
         'branded', tol_branded, False),

        ('v5: Same product, unknown pack on both sides → SHOULD match',
         _mk('baked beans tomato sauce', 415, 'g', 'Tesco', brand='heinz', pq=None),
         _mk('baked beans tomato sauce', 415, 'g', 'ASDA',  brand='heinz', pq=None),
         'branded', tol_branded, True),

        ('v5: Same product, explicit single pack on both sides → SHOULD match',
         _mk('baked beans tomato sauce', 415, 'g', 'Tesco', brand='heinz', pq=1),
         _mk('baked beans tomato sauce', 415, 'g', 'ASDA',  brand='heinz', pq=1),
         'branded', tol_branded, True),

        ('v5: Single pack vs NaN → SHOULD match (known single is OK with unknown)',
         _mk('baked beans tomato sauce', 415, 'g', 'Tesco', brand='heinz', pq=1),
         _mk('baked beans tomato sauce', 415, 'g', 'ASDA',  brand='heinz', pq=None),
         'branded', tol_branded, True),

        # ── v5.1 audit-fix tests ──────────────────────────────────────────────
        ('v5.1: Lychee lemonade vs Elderflower lemonade → must NOT match (flavor: lychee)',
         _mk('lychee lemonade',      250, 'ml', 'Tesco', brand=None, ptype='own_brand'),
         _mk('elderflower lemonade', 250, 'ml', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        ('v5.1: Pesto basil vs Pesto red → must NOT match (flavor: basil)',
         _mk('basil pesto', 190, 'g', 'Tesco', brand=None, ptype='own_brand'),
         _mk('red pesto',   190, 'g', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        ('v5.1: Reduced fat hummus vs plain hummus → must NOT match (one-sided: reduced)',
         _mk('reduced fat hummus', 200, 'g', 'Tesco', brand=None, ptype='own_brand'),
         _mk('hummus',             200, 'g', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        ('v5.1: Chocolate buttons vs chocolate block → must NOT match (one-sided: buttons)',
         _mk('milk chocolate buttons', 100, 'g', 'Tesco', brand=None, ptype='own_brand'),
         _mk('milk chocolate',         100, 'g', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        ('v5.1: Peaches in juice vs peaches in syrup → must NOT match (prep conflict)',
         _mk('peach slices in juice',  410, 'g', 'Tesco', brand=None, ptype='own_brand'),
         _mk('peach slices in syrup',  410, 'g', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        ('v5.1: Ricotta pasta vs Mascarpone pasta → must NOT match (filling conflict)',
         _mk('ricotta spinach tortelloni',    300, 'g', 'Tesco', brand=None, ptype='own_brand'),
         _mk('mascarpone mushroom tortelloni', 300, 'g', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        # ── v5.2 pack-normalised unit comparison tests ────────────────────────
        # Both unit_values are TOTAL weight (after normalise.py fix), so direct
        # unit comparison succeeds (360g ≈ 360g) without needing the per-pack division.
        ('v5.2: 10-pack 360g total vs 10-pack 360g total → SHOULD match',
         _mk('oat so simple golden syrup porridge',         360.0, 'g', 'ASDA',  brand='quaker', pq=10),
         _mk('oat so simple golden syrup porridge sachets', 360.0, 'g', 'Sains', brand='quaker', pq=10),
         'branded', tol_branded, True),

        ('v5.2: ASDA 2-pack 250g total vs Sains 1-pack 125g → must NOT match (different pack count)',
         _mk('creme egg', 250.0, 'g', 'ASDA',  brand='cadbury', pq=2),
         _mk('creme egg', 125.0, 'g', 'Sains', brand='cadbury', pq=1),
         'branded', tol_branded, False),

        # ── v5.2 new hard-conflict tests ──────────────────────────────────────
        ('v5.2: Classic Banana porridge vs Golden Syrup porridge → must NOT match (banana vs syrup)',
         _mk('oat so simple classic banana porridge sachets', 360.0, 'g', 'Sains', brand='quaker', pq=10),
         _mk('oat so simple golden syrup porridge',           360.0, 'g', 'ASDA',  brand='quaker', pq=10),
         'branded', tol_branded, False),

        ('v5.2: Light soft cheese vs Chives soft cheese → must NOT match (one-sided: light)',
         _mk('light soft cream cheese',  150, 'g', 'Tesco', brand=None, ptype='own_brand'),
         _mk('chives soft cream cheese', 150, 'g', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        ('v5.2: Beer cans vs beer bottle → must NOT match (packaging: can vs bottle)',
         _mk('premium lager beer cans',   440, 'ml', 'Tesco', brand='stella artois'),
         _mk('premium lager beer bottle', 440, 'ml', 'ASDA',  brand='stella artois'),
         'branded', tol_branded, False),

        ('v5.2: Beer cans vs beer cans → SHOULD match (packaging: same)',
         _mk('premium lager beer cans', 440, 'ml', 'Tesco', brand='stella artois'),
         _mk('premium lager beer cans', 440, 'ml', 'ASDA',  brand='stella artois'),
         'branded', tol_branded, True),

        ('v5.2: Decaff instant coffee vs regular → must NOT match (one-sided: decaff)',
         _mk('original decaff instant coffee', 200, 'g', 'Tesco', brand='nescafe'),
         _mk('original instant coffee',        200, 'g', 'ASDA',  brand='nescafe'),
         'branded', tol_branded, False),

        # ── v6: ingredient variety conflict tests ─────────────────────────────
        # These verify that meat-type tokens (lamb, ham, chicken, beef, pork, …)
        # are now in FLAVOR_NAMED_TOKENS and caught by _hard_conflict_check BEFORE
        # SHORT_STRIPPED_THRESHOLD is applied (the score is ~0.91 which was previously
        # caught only by the 0.92 threshold — now it is stopped by hard conflict).
        ('v6: Knorr chicken vs beef stock cubes → must NOT match (ingredient conflict)',
         _mk('stock cubes chicken', 104, 'g', 'Tesco', brand='knorr'),
         _mk('stock cubes beef',    104, 'g', 'ASDA',  brand='knorr'),
         'branded', tol_branded, False),

        ('v6: Pork sausages vs chicken sausages (own-brand) → must NOT match (ingredient)',
         _mk('pork sausages',    400, 'g', 'Tesco', brand=None, ptype='own_brand'),
         _mk('chicken sausages', 400, 'g', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        ('v6: Lamb mince vs beef mince (own-brand) → must NOT match (ingredient)',
         _mk('lean lamb mince', 500, 'g', 'Tesco', brand=None, ptype='own_brand'),
         _mk('lean beef mince', 500, 'g', 'ASDA',  brand=None, ptype='own_brand'),
         'own_brand', tol_own, False),

        # ── v6: Sainsbury's own-brand matching (post brand-strip fix) ─────────
        # After fixing SUPERMARKET_BRANDS key 'Sainsburys' → 'Sains', the Sains
        # brand prefix is stripped, tier_type is set to 'standard', and
        # core_product_name no longer contains the supermarket prefix.
        ('v6: Sains semi-skimmed milk vs ASDA semi-skimmed milk → SHOULD match',
         _mk('semi skimmed milk', 2000, 'ml', 'Sains', brand=None, tier='standard', ptype='own_brand'),
         _mk('semi skimmed milk', 2000, 'ml', 'ASDA',  brand=None, tier='standard', ptype='own_brand'),
         'own_brand', tol_own, True),

        ('v6: Sains butter salted vs ASDA butter salted → SHOULD match',
         _mk('butter salted', 250, 'g', 'Sains', brand=None, tier='standard', ptype='own_brand'),
         _mk('butter salted', 250, 'g', 'ASDA',  brand=None, tier='standard', ptype='own_brand'),
         'own_brand', tol_own, True),

        ('v6: Sains standard milk vs Morrisons premium tier milk → must NOT match (tier mismatch)',
         _mk('semi skimmed milk', 2000, 'ml', 'Sains',     brand=None, tier='standard', ptype='own_brand'),
         _mk('semi skimmed milk', 2000, 'ml', 'Morrisons', brand=None, tier='premium',  ptype='own_brand'),
         'own_brand', tol_own, False),

        # ── v6: multipack total-weight consistency ────────────────────────────
        # After normalise.py fix, "5×19.9g" → unit_value=99.5g (total), pack_qty=5
        # for ALL stores.  Direct unit comparison: 99.5g ≈ 99.5g → match.
        ('v6: 5-pack 99.5g total vs 5-pack 99.5g total → SHOULD match (multipack total-weight)',
         _mk('animals chocolate biscuits', 99.5, 'g', 'ASDA',  brand='cadbury', pq=5),
         _mk('animals chocolate biscuits', 99.5, 'g', 'Sains', brand='cadbury', pq=5),
         'branded', tol_branded, True),

        ('v6: 5-pack 99.5g total vs 1-pack 19.9g → must NOT match (pack count mismatch)',
         _mk('animals chocolate biscuits', 99.5, 'g', 'ASDA',  brand='cadbury', pq=5),
         _mk('animals chocolate biscuits', 19.9, 'g', 'Sains', brand='cadbury', pq=1),
         'branded', tol_branded, False),
    ]


def run_tests(compute_similarity_fn, tol_branded=0.05, tol_own=0.05, tol_unbranded=0.05,
              verbose=True):
    """Run all test cases against compute_similarity_fn.

    Returns True if all tests pass, False otherwise.
    """
    cases = build_test_cases(tol_branded, tol_own, tol_unbranded)
    all_pass = True
    for desc, ra, rb, ptype, tol, expected in cases:
        match, score = compute_similarity_fn(ra, rb, ptype, tol)
        ok = (match == expected)
        if not ok:
            all_pass = False
        if verbose:
            status = '✓' if ok else '✗ FAIL'
            print(f'  {status}  {desc}: match={match}, score={score:.3f}')
    return all_pass


if __name__ == '__main__':
    # Standalone mode: import compute_similarity from clustering_v5 by loading
    # only the module-level definitions (not the main clustering run).
    # clustering_v5.py guards its main data-loading body with:
    #   if not globals().get('CLUSTERING_TEST_IMPORT', False):
    # which is set below before exec_module is called.
    import os
    import importlib.util
    import types

    _here = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        'clustering_v5',
        os.path.join(_here, 'clustering_v5.py'),
    )
    _mod = types.ModuleType(_spec.name)
    _mod.CLUSTERING_TEST_IMPORT = True   # prevents data-loading body from running
    sys.modules[_spec.name] = _mod       # register so __name__ is correct inside module
    try:
        _spec.loader.exec_module(_mod)
    except SystemExit:
        pass  # expected: module raises SystemExit(0) after function definitions to skip clustering

    print('\nRunning ALL similarity self-tests (standalone)...')
    _all_pass = run_tests(
        _mod.compute_similarity,
        tol_branded=_mod.UNIT_TOLERANCE_BRANDED,
        tol_own=_mod.UNIT_TOLERANCE_OWN_BRAND,
        tol_unbranded=_mod.UNIT_TOLERANCE_UNBRANDED,
    )
    print(f'\nAll tests passed: {_all_pass}')
    sys.exit(0 if _all_pass else 1)
