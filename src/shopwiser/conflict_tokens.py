"""Single source of truth for hard-conflict vocabulary.

Both the rule-based clustering pipeline (``shopwiser.rule_matcher``) and the ML
matching pipeline (``shopwiser.ml_matcher``) consume the same vocabulary
from here. Edit in one place — both pipelines pick the change up on their
next run.

Token sets
----------
HARD_CONFLICT_NORM
    Synonym normalisation applied BEFORE every set lookup. Maps spelling
    variants and plural forms onto a canonical singular (``raspberries`` →
    ``raspberry``) so the conflict checks fire regardless of how the retailer
    labelled the product.

FLAVOR_NAMED_TOKENS
    Named flavours, ingredient varieties and brand-line discriminators.
    The clash check fires only when BOTH products contain at least one named
    token AND the sets do not overlap (e.g. ``raspberry`` vs ``strawberry``).

ONE_SIDED_CONFLICT_TOKENS
    Tokens whose presence in exactly one product is itself a strong
    differentiator (e.g. ``decaf`` vs not, ``baby`` vs not). Includes the
    union of FLAVOR_NAMED_TOKENS so asymmetric flavour presence
    ("Pringles Original" vs "Pringles Cheese & Onion") is caught too.

MILK_BASE_TOKENS, COOKING_STATE_TOKENS, PACKAGING_FORMAT_TOKENS,
PREPARATION_CONFLICT_PAIRS
    Mutually-exclusive groups: a clash fires when each side picks a different
    token from the same group.

check_hard_conflict
    The matching pipelines' shared rejection function — returns True when any
    of the gates above flags a hard conflict between two product names.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Synonym / spelling normalisation (applied before every set lookup)
# ---------------------------------------------------------------------------

# Maps any token in this dict to its canonical form. Use to fold spelling
# variants and common plurals so the conflict checks below fire regardless
# of how a retailer labelled the product.
HARD_CONFLICT_NORM: dict[str, str] = {
    # Spelling / abbreviation variants
    "soya": "soy",
    "roasted": "roast",
    "decaffeinated": "decaf",
    "decaff": "decaf",
    # Packaging singular/plural
    "cans": "can",
    "bottles": "bottle",
    # Berry / fruit plurals → singular (catches "Strawberries" ↔ "Raspberry" etc.)
    "raspberries": "raspberry",
    "strawberries": "strawberry",
    "blueberries": "blueberry",
    "blackberries": "blackberry",
    "cranberries": "cranberry",
    "cherries": "cherry",
    "grapes": "grape",
    "peaches": "peach",
    "apricots": "apricot",
    "apples": "apple",
    "pears": "pear",
    "plums": "plum",
    "limes": "lime",
    "lemons": "lemon",
    "oranges": "orange",
    "bananas": "banana",
    "mangoes": "mango",
    "pineapples": "pineapple",
    # Vegetable plurals
    "onions": "onion",
    "tomatoes": "tomato",
    "potatoes": "potato",
    "carrots": "carrot",
    "mushrooms": "mushroom",
    "peppers": "pepper",
    "cucumbers": "cucumber",
    # Meat / protein plurals
    "prawns": "prawn",
    "shrimps": "shrimp",
    "crabs": "crab",
    "sausages": "sausage",
    "burgers": "burger",
    "bulgar": "bulgur",
    "macaroni": "pasta",
    "spaghetti": "pasta",
    # BBQ / barbecue spelling variants (all → canonical "bbq")
    "barbecue": "bbq",
    "barbeque": "bbq",
}


# ---------------------------------------------------------------------------
# Named flavour / variety tokens
# ---------------------------------------------------------------------------

# Clash fires when BOTH products carry at least one named token AND the sets
# do not overlap. This catches "Strawberry yogurt" vs "Raspberry yogurt"
# even when surface similarity is high.
FLAVOR_NAMED_TOKENS: frozenset[str] = frozenset({
    # Fruit / herb / spice flavours
    "ginger", "mint", "raspberry", "lemon", "orange", "cherry",
    "strawberry", "blueberry", "mango", "blackcurrant", "blackberry",
    "elderflower", "rhubarb", "lime", "peach", "apricot", "vanilla",
    "caramel", "toffee", "honey", "maple", "cinnamon", "cola",
    "lychee", "basil", "chilli", "banana", "syrup",
    "kiwi", "berry", "melon", "grape", "pear", "pineapple",
    "pomegranate", "watermelon", "passion", "fig", "plum",
    "cranberry", "apple",
    # Confectionery / dessert variants
    "peppermint", "spearmint", "honeycomb", "fudge", "praline",
    "marzipan", "nougat",
    # Meat / poultry / seafood — different protein = different product
    "lamb", "ham", "pork", "beef", "chicken", "turkey", "duck",
    "venison", "bacon",
    "salmon", "tuna", "cod", "haddock", "prawn", "shrimp", "crab",
    "mackerel", "trout", "sardine", "anchovy",
    # Italian coffee-range / brand variant tokens
    "rossa", "oro", "intenso", "americano",
    # Snack / sauce / curry / cheese flavour discriminators
    "paprika", "vinegar", "pickled", "marmite", "worcester",
    "ketchup", "mustard", "horseradish", "wasabi",
    "cheddar", "parmesan", "mozzarella", "feta", "halloumi",
    "tikka", "korma", "masala", "jalfrezi", "madras", "vindaloo",
    "rogan", "biryani", "pad", "thai", "szechuan", "teriyaki",
    "hoisin", "satay", "katsu",
    "pesto", "arrabbiata", "carbonara", "bolognese", "lasagne",
    "jerk", "cajun", "creole", "piri",
    "cocktail", "salad", "ranch",
    # Snack flavour discriminators — BBQ is a distinct flavour variant
    "bbq",
    # Wine grape varieties — distinct varietals never substitute
    "chardonnay", "cabernet", "sauvignon", "merlot", "pinot",
    "shiraz", "malbec", "riesling", "zinfandel", "prosecco",
    "champagne", "chenin", "viognier", "gewurztraminer",
    "tempranillo", "sangiovese",
})


# ---------------------------------------------------------------------------
# Mutually-exclusive groups (clash on different members)
# ---------------------------------------------------------------------------

MILK_BASE_TOKENS: frozenset[str] = frozenset({
    "soy", "oat", "almond", "hazelnut", "cashew", "rice", "coconut",
    "hemp", "pea", "macadamia", "pistachio",
    "ricotta", "mascarpone",
})

COOKING_STATE_TOKENS: frozenset[str] = frozenset({
    "raw", "roast", "smoked", "unsmoked", "dried", "cured",
})

SPICE_LEVEL_TOKENS: frozenset[str] = frozenset({
    "mild", "medium", "hot", "spicy", "extra", "inferno",
})

PACKAGING_FORMAT_TOKENS: frozenset[str] = frozenset({"can", "bottle"})

# Each frozenset is a mutually-exclusive group: a clash fires when each
# side picks a different token from the same group.
PREPARATION_CONFLICT_PAIRS: list[frozenset[str]] = [
    frozenset({"juice", "syrup"}),
    # Preservation / cooking medium: "sardines in sunflower oil" ≠ "sardines in tomato sauce"
    frozenset({"oil", "sauce"}),
]


# ---------------------------------------------------------------------------
# One-sided tokens (presence asymmetry = clash)
# ---------------------------------------------------------------------------

# Tokens that, present in exactly one product, indicate different variants.
# The set is checked AFTER HARD_CONFLICT_NORM so synonyms collapse first.
ONE_SIDED_CONFLICT_TOKENS: frozenset[str] = frozenset(set({
    # Dietary / preparation / format
    "baby", "reduced", "decaf", "vegan", "organic",
    "wholemeal", "wholegrain", "skimmed", "lite", "light", "zero",
    # Bread variants
    "granary", "seeded", "sourdough", "multigrain",
    # Product-type discriminators
    "soup", "jam", "juice", "buttons", "pate",
    # Drink / wine variants
    "rose", "blonde",
    # Misc
    "eyed",
    # Nut & confectionery variants
    "almond", "hazelnut", "pistachio", "walnut", "pecan", "coconut",
    "salted", "unsalted",
    # Chocolate / bread colour variants
    "white", "dark",
    # Indian cuisine style markers (also in FLAVOR for the both-have case)
    "tikka", "masala", "jalfrezi", "korma", "vindaloo", "madras",
    "balti", "biryani", "pilau", "tandoori", "gujarati", "bhuna", "rogan",
    # Smoking / curing
    "smoked", "smoky",
    # Format markers — kits / mixes / sliced are different products from the
    # ready-made counterpart.
    "kit", "mix", "pie", "dinner", "meal", "fillet", "fillets", "chunks", "mince", 
    "diced", "meatballs", "burger", "burgers", "sausage", "sausages", "bites", 
    "pastilles", "randoms", "halves", "slices", "pieces", "whole", "chopped", 
    "puree", "passata", "noodles", "pasta", "spaghetti", "macaroni", "penne", "fusilli",
    "gluten", "dairy", "wheat", "soya", "plant-based", "vegetarian", "vegan", "meat-free",
    # Protein/dish format
    "quiche", "moussaka", "shanks", "kebab", "goujons", "nuggets", "fishcakes",
    # Rice variety
    "basmati", "jasmine",
    # Confectionery / Easter variants
    "sherbet", "sherbets", "bunny",
    # Snack / sauce variant tokens — extends FLAVOR with the asymmetric case
    "sour", "paprika", "vinegar", "pickled", "marmite", "worcester",
    "ketchup", "mustard", "horseradish", "wasabi",
    "cheddar", "parmesan", "mozzarella", "feta", "halloumi",
    "carbonara", "bolognese", "pesto", "arrabbiata", "lasagne",
    "teriyaki", "hoisin", "satay", "katsu", "jerk", "cajun", "creole", "piri",
    "garlic", "rosemary", "thyme", "oregano", "parsley", "coriander", "basil", "sage", "tarragon", "mint", "chive", "chives",
    "concentrate", "concentrated", "remix", "edition", "limited",
    "strong", "softmints", "mints",
    "loops", "pops", "hoops", "rings", "balls", "squares", "shapes", "stars",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    # Wine grape varieties — one-sided presence is a strong differentiator
    # (e.g. "Chardonnay" vs plain "White Wine")
    "chardonnay", "cabernet", "sauvignon", "merlot", "pinot",
    "shiraz", "malbec", "riesling", "prosecco", "champagne",
    "tempranillo", "sangiovese",
    # Pastry type — puff and shortcrust are different products
    "puff", "shortcrust",
    # BBQ is a distinct flavour variant (Pringles BBQ vs Cheese & Onion)
    "bbq",
    # Bean / legume variety — different beans are different products
    "cannellini", "haricot", "kidney", "borlotti", "flageolet", "edamame",
}) | FLAVOR_NAMED_TOKENS)
# Including the union of FLAVOR_NAMED_TOKENS catches asymmetric flavour
# presence ("Pringles Original" vs "Pringles Cheese & Onion").

# Intentionally NOT in any conflict set:
#   "hot", "sweet", "spicy"  — too broad, clash with brand names like
#                              "Sweet Freedom" and use-case words like
#                              "Hot Chocolate".


# ---------------------------------------------------------------------------
# Tokenisation + the shared check
# ---------------------------------------------------------------------------

def _tokenise(name: str) -> tuple[set[str], set[str]]:
    """Lowercase whitespace tokens of *name* + the same set after applying
    HARD_CONFLICT_NORM. Returns ``(raw_tokens, normalised_tokens)``."""
    raw = set(str(name).lower().split())
    norm = {HARD_CONFLICT_NORM.get(t, t) for t in raw}
    return raw, norm


# Phrase-level patterns that individual token checks can't catch.
# These are checked against RAW product names (before normalisation strips them).
_NAS_RE = re.compile(r'\bno\s+added\s+sugar\b', re.I)


def check_hard_conflict(name_a: str, name_b: str) -> bool:
    """True when any hard-conflict gate fires between the two product names.

    Gates (in order of evaluation):
      1. Milk-base mismatch (oat vs almond …)
      2. Named flavour clash (BOTH have named tokens, no overlap)
      3. Cooking-state mismatch (smoked vs unsmoked …)
      4. One-sided token (decaf in one but not the other)
      5. Mutually-exclusive preparation pair (in juice vs in syrup …)
      6. Packaging format mismatch (can vs bottle)

    Note: "No Added Sugar" phrase conflicts require raw names — use
    ``check_phrase_conflict`` on the original product names alongside this.
    """
    raw_a, toks_a = _tokenise(name_a)
    raw_b, toks_b = _tokenise(name_b)

    # 1. Milk base
    base_a = toks_a & MILK_BASE_TOKENS
    base_b = toks_b & MILK_BASE_TOKENS
    if base_a and base_b and not (base_a & base_b):
        return True

    # 2. Named flavour clash
    flav_a = toks_a & FLAVOR_NAMED_TOKENS
    flav_b = toks_b & FLAVOR_NAMED_TOKENS
    if flav_a and flav_b and not (flav_a & flav_b):
        return True

    # 3. Cooking state
    state_a = toks_a & COOKING_STATE_TOKENS
    state_b = toks_b & COOKING_STATE_TOKENS
    if state_a and state_b and not (state_a & state_b):
        return True

    # Spice level
    spice_a = toks_a & SPICE_LEVEL_TOKENS
    spice_b = toks_b & SPICE_LEVEL_TOKENS
    if spice_a and spice_b and not (spice_a & spice_b):
        return True

    # 4. One-sided
    for tok in ONE_SIDED_CONFLICT_TOKENS:
        if (tok in toks_a) != (tok in toks_b):
            return True

    # 5. Preparation pair
    for pair in PREPARATION_CONFLICT_PAIRS:
        hits_a = raw_a & pair
        hits_b = raw_b & pair
        if hits_a and hits_b and not (hits_a & hits_b):
            return True

    # 6. Packaging
    pkg_a = toks_a & PACKAGING_FORMAT_TOKENS
    pkg_b = toks_b & PACKAGING_FORMAT_TOKENS
    if pkg_a and pkg_b and not (pkg_a & pkg_b):
        return True

    return False


def check_phrase_conflict(raw_name_a: str, raw_name_b: str) -> bool:
    """Catch conflicts visible only in raw product names that normalisation strips.

    Use this alongside ``check_hard_conflict`` on the *normalised* names when
    the raw names are available.  The checks here are phrase-level patterns that
    cannot be expressed as individual token lookups.

    Currently checks:
      - "No Added Sugar" asymmetry: if exactly one side carries this phrase,
        the products are different dietary variants.
    """
    has_nas_a = bool(_NAS_RE.search(raw_name_a))
    has_nas_b = bool(_NAS_RE.search(raw_name_b))
    if has_nas_a != has_nas_b:
        return True
    return False
