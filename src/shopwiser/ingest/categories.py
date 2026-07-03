"""Coarse category assignment for scraped products.

The matcher blocks on ``cat_norm`` (see rule_matcher.data_prep) and hard-gates
frozen vs non-frozen, so *every* ingested row needs one of the six coarse
categories used by the original dataset:

    food_cupboard, fresh_food, drinks, frozen, free-from, bakery

Coverage of the scraper's fine-grained ``details_category`` is very uneven
(ASDA 100%, Tesco ~55%, Sainsbury's ~0%), so classification is two-tier:

1. ``CATEGORY_RULES`` against the retailer section name when present
   ("Ice Cream Tubs" → frozen, "Red Wine" → drinks);
2. ``NAME_RULES`` against the product title otherwise — deliberately the same
   bucket vocabulary so the same product lands in the same bucket whichever
   retailer it came from.

Anything unmatched falls back to ``food_cupboard`` (the majority bucket in the
original dataset). A wrong-but-consistent bucket costs nothing; an
inconsistent one costs a candidate pair, which the cross-category blocking
keys in the rule matcher partially recover.

Rules are ordered: first hit wins. Name rules put bakery before drinks so
"Tea Cakes" / "Coffee Cake" resolve to bakery, not drinks.
"""

import re
from typing import List, Optional, Tuple

FOOD_CUPBOARD = 'food_cupboard'
FRESH_FOOD = 'fresh_food'
DRINKS = 'drinks'
FROZEN = 'frozen'
FREE_FROM = 'free-from'   # hyphenated, as in the original raw.csv
BAKERY = 'bakery'

# (bucket, include pattern, exclude pattern or None)
_Rule = Tuple[str, 're.Pattern[str]', Optional['re.Pattern[str]']]


def _rule(bucket: str, include: str, exclude: str = None) -> _Rule:
    return (
        bucket,
        re.compile(include, re.IGNORECASE),
        re.compile(exclude, re.IGNORECASE) if exclude else None,
    )


# ── Tier 1: retailer section names (fine-grained, low ambiguity) ─────────────
CATEGORY_RULES: List[_Rule] = [
    _rule(FREE_FROM, r'free[\s-]*from|gluten[\s-]*free|dairy[\s-]*free|lactose[\s-]*free|lactofree'),
    _rule(FROZEN, r'frozen|ice\s*cream|ice\s*loll|sorbet|gelato'),
    _rule(BAKERY,
          r'bakery|\bbread\b|\brolls?\b|baguette|bagel|croissant|crumpet|brioche'
          r'|pastries|doughnut|donut|muffin|scone|teacake|pitta|naan|wraps?\s*(?:&|and)?\s*(?:tortillas|thins)'
          r'|\bcakes?\b',
          exclude=r'rice\s*cakes?|fish\s*cakes?|cake\s*decorat'),
    _rule(DRINKS,
          r'\bwines?\b|\bbeers?\b|lager|\bcider\b|\bales?\b|\bstout\b|\bipas?\b|spirits?'
          r'|whisky|whiskey|vodka|\bgin\b|\brum\b|brandy|liqueur|cocktail|prosecco|champagne|sherry'
          r'|juice|smoothie|squash|cordial|\bwater\b|\bcola\b|lemonade|energy\s*drink|sports?\s*drink'
          r'|mixers?|tonic|\btea\b|\bteas\b|coffee|hot\s*chocolate|milkshake|kombucha|\bdrinks?\b',
          exclude=r'vinegar|cooking\s*wine'),
    _rule(FRESH_FOOD,
          r'fruit|\bveg(?:etables?)?\b|salad|potato|tomato|grape|berr(?:y|ies)|banana|apple|pear'
          r'|citrus|orange|lemon|lime|melon|cherries'
          r'|\bmeat\b|chicken|beef|pork|lamb|turkey|duck|sausage|bacon|\bham\b'
          r'|fish|seafood|salmon|prawn|deli\b|delicatessen'
          r'|\bmilk\b|cheese|yogurt|yoghurt|butter|\beggs?\b|\bcream\b|chilled|ready\s*meals?|pizza'
          r'|desserts?',
          exclude=r'condensed|evaporated|uht|long\s*life|peanut\s*butter|tinn?ed|canned'),
    # everything else — tins, jars, snacks, confectionery, baking, world foods…
]

# ── Tier 2: product titles (higher ambiguity → tighter guards) ───────────────
NAME_RULES: List[_Rule] = [
    _rule(FREE_FROM, r'free[\s-]*from|gluten[\s-]*free|dairy[\s-]*free|lactose[\s-]*free|lactofree'),
    _rule(FROZEN, r'\bfrozen\b|ice\s*cream|ice\s*loll|sorbet|gelato|choc\s*ices?\b'),
    _rule(BAKERY,
          r'\bbread\b|\bloa(?:f|ves)\b|baguette|bagel|croissant|crumpet|brioche|pain\s+au'
          r'|doughnut|donut|muffin|scone|teacakes?|hot\s*cross\s*bun|\bbuns?\b|ciabatta|sourdough'
          r'|bloomer|focaccia|panini|croquette|pancakes?|crepes?|waffles?'
          r'|(?<!sausage )(?<!spring )(?<!seaweed )(?<!fruit )\brolls?\b'
          r'|(?<!fish )(?<!rice )(?<!crab )(?<!potato )\bcakes?\b',
          # baking-aisle mixes/kits/toppers are cupboard, not the baked good
          exclude=r'toilet|kitchen\s*roll|\bmix\b|\bkit\b|decorat|toppers?|\bcases?\b|candles?'),
    _rule(DRINKS,
          r'\bwine\b|prosecco|champagne|\blager\b|\bbeer\b|\bcider\b|\bale\b|\bstout\b'
          r'|whisky|whiskey|vodka|\bgin\b|\brum\b|brandy|liqueur|tequila|sangria|\bport\b|sherry'
          r'|cocktail|mocktail|martini|mojito|margarita|\bmimosa\b|aperitif|vermouth|schnapps'
          r'|absinthe|\bsake\b|\bmead\b|\bperry\b|shandy|aperol|campari|amaretto|\bbaijiu\b'
          r'|premixed|pre[\s-]*mixed|\bmixer\b'
          # grape varieties / wine styles that often appear without the word "wine"
          r'|shiraz|\bsyrah\b|merlot|malbec|cabernet|sauvignon|chardonnay|pinot\b|\brioja\b'
          r'|chianti|zinfandel|grenache|tempranillo|prosecco|chablis|sancerre|riesling'
          r'|pinotage|viognier|albari|verdejo|montepulciano|valpolicella|beaujolais'
          r'|\bbordeaux\b|burgundy|\brosé\b|ros[eé]\s*wine|\bcava\b|\bsoave\b|gavi\b'
          r'|\bjuice\b|smoothie|squash|cordial|\bcola\b|lemonade|energy\s*drink|sports?\s*drink'
          r'|tonic\s*water|still\s*water|sparkling\s*water|mineral\s*water|spring\s*water'
          r'|coconut\s*water|flavoured\s*water'
          r'|soya\s*drink|\boat\s*drink|almond\s*drink|rice\s*drink|kombucha|iced\s*tea'
          r'|milkshake|\btea\s*bags?\b|\btea\b|coffee|espresso|hot\s*chocolate|cocoa\s*drink',
          exclude=r'vinegar|batter|butternut|gravy|marinade|\bstock\b|wine\s*gums?|sauce'
                  r'|brandy\s*butter|tea\s*towel|tea\s*cake|coffee\s*cake|rum\s*baba|sake\s*mirin'),
    _rule(FRESH_FOOD,
          r'\bmilk\b|cheese|yogurt|yoghurt|\bbutter\b|\beggs?\b|\bcream\b'
          r'|chicken|beef|pork|lamb|turkey|duck|sausage|bacon|\bham\b|mince|steak|fillet'
          r'|salmon|\bcod\b|prawn|haddock|mackerel|seafood|\bfish\b'
          r'|banana|\bgrapes?\b|strawberr|raspberr|blueberr|cherries|peaches?|nectarine|plums?'
          r'|apples?\b|pears?\b|melon|pineapple|mango(?:es)?\b|kiwi|avocado|citrus'
          r'|potato(?:es)?\b|tomato(?:es)?\b|salad|lettuce|cucumber|broccoli|carrots?\b|onions?\b'
          r'|peppers\b|mushrooms?|spinach|cabbage|cauliflower|courgette|leeks?\b|celery'
          r'|ready\s*meal|pizza|quiche|coleslaw|houmous|hummus|dips?\b|trifle|sandwich',
          # Ambient/confectionery/preserve terms that borrow fresh vocabulary
          # (a lone fruit word must not pull jam/juice/candy/cereal-bars into fresh).
          exclude=r'chocolate|coconut\s*milk|condensed|evaporated|milk\s*powder|formula'
                  r'|peanut\s*butter|almond\s*butter|cashew\s*butter|cocoa\s*butter'
                  r'|easter|kinder|crisps|snacks?\b|ketchup|puree|passata|chutney|pickle'
                  r'|chopped\s*tomatoes|plum\s*tomatoes|sun[\s-]*dried|tinn?ed|canned|soup'
                  r'|noodles?|pot\b|sauce|seasoning|gravy|paste'
                  r'|conserve|\bjam\b|marmalade|\bjelly\b|compote|\bcurd\b|spread|relish'
                  r'|dried|candied|\bcandy\b|\bsweets\b|\bgums?\b|pastilles?|\bbar\b|\bbars\b'
                  r'|cereal|granola|muesli|oat\s*(?:bar|stick|bite)|flour|\bcake\b|biscuits?'
                  r'|\bjerky\b|\bcrackers?\b|popcorn|\bteabags?\b|squash|cordial|\bjuice\b'
                  r'|\bmix\b|\bkit\b|decorat'
                  r'|yoghurt[\s-]*coated|yogurt[\s-]*coated|infusion'),
    # everything else → food_cupboard
]


def _apply(rules: List[_Rule], text: str) -> Optional[str]:
    for bucket, include, exclude in rules:
        if include.search(text) and not (exclude and exclude.search(text)):
            return bucket
    return None


def classify(details_category, product_name) -> str:
    """Coarse bucket for a scraped row; falls back to ``food_cupboard``."""
    if isinstance(details_category, str) and details_category.strip():
        hit = _apply(CATEGORY_RULES, details_category)
        if hit:
            return hit
        # Section name exists but matched nothing (e.g. "South Asian",
        # "Spices") — those sections are genuinely cupboard territory more
        # often than the title is wrong, but let an explicit title signal
        # (frozen/drinks/…) override before defaulting.
        return _apply(NAME_RULES, str(product_name or '')) or FOOD_CUPBOARD
    return _apply(NAME_RULES, str(product_name or '')) or FOOD_CUPBOARD
