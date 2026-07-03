"""Tests for the scraped-catalogue ingestion (price/unit parsing, name, own-brand)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from shopwiser.ingest.categories import classify
from shopwiser.ingest.main import build_name, detect_own_brand, _own_label_prefixes
from shopwiser.ingest.parsing import parse_price, parse_unit_price


def test_parse_price():
    assert parse_price('£2.15') == 2.15
    assert parse_price('£1,099.95') == 1099.95
    assert parse_price('£4.9') == 4.9
    assert parse_price('Any 4 for 3') is None
    assert parse_price('£1.50 / kg') is None  # unit price misfiled in price column
    assert parse_price(None) is None


def test_parse_unit_price_fixed_denoms():
    assert parse_unit_price('£14.93/KG') == (14.93, 'kg')
    assert parse_unit_price('£1.08/litre') == (1.08, 'l')
    assert parse_unit_price('£3.90 / ltr') == (3.9, 'l')
    assert parse_unit_price('23.8p/EA') == (0.238, 'unit')
    assert parse_unit_price('£12.00/litre') == (12.0, 'l')


def test_parse_unit_price_quantified_denoms():
    # per-100g must become per-kg (×10)
    assert parse_unit_price('87.0p/100g') == (8.7, 'kg')
    # per-75cl bottle → per-litre
    assert parse_unit_price('£10.67/75cl') == (round(10.67 / 75 * 100, 4), 'l')


def test_parse_unit_price_edge_cases():
    assert parse_unit_price('£11.80/kg DR.WT') == (11.8, 'kg')   # trailing annotation
    assert parse_unit_price('.5p/EA') == (0.005, 'unit')          # leading-dot pence
    assert parse_unit_price('garbage') == (None, None)
    assert parse_unit_price(None) == (None, None)


def test_build_name_prepends_brand():
    # ASDA titles omit the brand; it lives in brand_name
    assert build_name('Tropical Granola 1kg', 'ASDA') == 'ASDA Tropical Granola 1kg'
    # already-prefixed titles are untouched
    assert build_name('ASDA Beef Lasagne 800g', 'ASDA') == 'ASDA Beef Lasagne 800g'
    assert build_name('Red Bull Energy 250ml', None) == 'Red Bull Energy 250ml'


def test_detect_own_brand():
    asda = _own_label_prefixes('ASDA')
    sains = _own_label_prefixes('Sains')
    assert detect_own_brand('ASDA Beef Lasagne 800g', 'ASDA', 'ASDA', asda)
    assert detect_own_brand("Sainsbury's Basmati Rice", None, 'Sains', sains)
    assert not detect_own_brand('Red Bull Energy 250ml', 'Red Bull', 'ASDA', asda)
    # brand_name carries the retailer even when the title does not start with it
    assert detect_own_brand('Extra Special Sourdough', 'Exceptional by ASDA', 'ASDA', asda)


def test_classify_buckets():
    assert classify('Ice Cream Tubs', 'Vanilla Ice Cream 1L') == 'frozen'
    assert classify('Red Wine', 'Merlot 75cl') == 'drinks'
    assert classify(None, 'Free From White Bloomer Loaf') == 'free-from'
    assert classify(None, 'Fresh Chicken Breast Fillets 650g') == 'fresh_food'
    assert classify(None, 'Baked Beans in Tomato Sauce 400g') == 'food_cupboard'
    # drink false-friends stay out of drinks
    assert classify(None, 'Wine Gums 500g') != 'drinks'
    assert classify(None, 'Fish Cakes 250g') != 'bakery'


def test_classify_false_friends():
    # confectionery / preserves borrowing fruit words must not become fresh_food
    assert classify(None, 'Raspberry Conserve 340g') == 'food_cupboard'
    assert classify(None, 'Nerds Candy Sweets Box Watermelon & Cherry 46.7g') == 'food_cupboard'
    assert classify(None, 'Tomato Relish 320g') == 'food_cupboard'
    # baking-aisle mixes/kits/decorations are cupboard, not the baked good
    assert classify(None, 'Betty Crocker Muffin Mix Kit 335g') == 'food_cupboard'
    assert classify(None, 'Unicorn Cake Decorations x12') == 'food_cupboard'
    assert classify(None, 'Victoria Sponge Cake') == 'bakery'   # a real cake still is bakery
    # cocktails and bare grape varieties are drinks
    assert classify(None, 'Funkin Cocktails Passion Fruit Martini 700ml') == 'drinks'
    assert classify(None, 'Jack Rabbit Shiraz 12 x 187ml') == 'drinks'
    assert classify(None, 'House Soave White Wine 225cl') == 'drinks'
    # guard: 'pinot' varietal pattern must not swallow 'pinto beans'
    assert classify(None, 'Pinto Beans 400g') == 'food_cupboard'
    # lactofree milk → free-from
    assert classify(None, 'Arla Lactofree Semi Skimmed Milk Drink 2L') == 'free-from'
