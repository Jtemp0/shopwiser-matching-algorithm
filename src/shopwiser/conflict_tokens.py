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
    # Nut plural → singular (so "peanuts" fires against FLAVOR token "peanut")
    "peanuts": "peanut",
    # Chocolate shorthand → canonical token
    "choc": "chocolate",   # "Choc Chip" → "chocolate chip"; FLAVOR ONE_SIDED then fires
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
    "cranberry", "apple", "tropical",
    # Nut / seed — distinct flavour protein bars, spreads, etc.
    "peanut", "almond",
    # Coffee / café drinks — mocha is a distinct beverage
    "mocha",
    # Sauce / condiment flavour discriminators
    "chipotle", "sriracha",
    # Confectionery / dessert variants
    "peppermint", "spearmint", "honeycomb", "fudge", "praline",
    "marzipan", "nougat", "crunchie",  # crunchie = honeycomb toffee inclusion (Cadbury)
    "ripple",   # Galaxy Ripple is a distinct product from plain Galaxy Milk Chocolate
    # Seed / grain variety — sesame bagels ≠ plain bagels (different bake)
    "sesame",
    # Meat / poultry / seafood — different protein = different product
    "lamb", "ham", "pork", "beef", "chicken", "turkey", "duck",
    "venison", "bacon", "chorizo",   # chorizo = distinct meat ingredient
    "salmon", "tuna", "cod", "haddock", "prawn", "shrimp", "crab",
    "mackerel", "trout", "sardine", "anchovy",
    # Coffee sub-types — distinct preparations are different products
    "latte", "espresso", "cappuccino", "macchiato", "lungo",
    # Italian coffee-range / brand variant tokens
    "rossa", "oro", "intenso", "americano", "millicano",
    # Cracker / snack brand-line discriminators
    "tuc",   # Jacob's TUC crackers ≠ Jacob's plain crackers (distinct product within same brand)
    # Honey variety — manuka is a certified specialty honey distinct from regular honey
    "manuka",
    # Oil seed variety — sunflower oil and rapeseed oil are distinct products
    "sunflower", "rapeseed",
    # Sauce flavour variant — buffalo is a distinct hot sauce flavour (not just heat level)
    "buffalo",
    # Geographic / regional flavour variant — Mediterranean Tonic ≠ plain Tonic Water
    "mediterranean",
    # Mango variety — kesar mango pulp is a distinct cultivar from generic mango pulp
    "kesar",
    # Tropical fruit as distinct flavour ingredient — papaya in a blend ≠ without papaya
    "papaya",
    # Lime / lychee are already present; guava and passionfruit added for completeness
    "guava", "passionfruit",
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
    # Vegetable / allium variants — onion is a primary flavour differentiator
    "onion",
    # Snack flavour discriminators — BBQ is a distinct flavour variant
    "bbq",
    # Confectionery sub-brand / product-line discriminators
    "flake",    # "Cadbury Flake Ice Cream" ≠ "Cadbury Creme Egg Ice Cream Tub"
    # Seed / grain variety
    "sesame",   # sesame bagels ≠ plain bagels
    # Generic seafood as a primary descriptor — asymmetric presence signals different product type
    "fish",     # "Fish Seasoning" ≠ "Caribbean Seasoning"; "Fish Soup" ≠ "Chicken Soup"
    # Pulse / legume as primary filling — lentil cottage pie ≠ plain cottage pie
    "lentil",
    # Cheese variety — stilton is a distinct flavoured soup/dish ingredient
    "stilton",
    # Dish variant — Chicken Kyiv is a distinct stuffed product vs plain chicken bites
    "kyiv",
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
    # Chocolate — distinct flavour/variant (Ambrosia Chocolate vs plain; Monster vs Monster Ultra)
    "chocolate",
    # Energy drink / soft drink variant markers
    "ultra", "max",
    # Olive oil grade — extra virgin vs regular are distinct products
    "virgin",
    # Cooking preparation / product sub-type markers
    "double",        # double strength, double cream vs single/regular
    "curry",         # chicken curry pie ≠ chicken pie
    "spiced",        # spiced ginger ≠ ginger
    "brewed",        # naturally brewed soy sauce ≠ regular soy sauce
    "naturally",     # naturally brewed / naturally sweetened — variant marker
    # Flavour/recipe discriminators not covered by FLAVOR_NAMED_TOKENS
    "spicy",         # "Spicy Sweet & Sour" ≠ "Sweet & Sour" (one-sided; SPICE_LEVEL needs BOTH)
    "hot",           # "Peperami Hot" ≠ "Peperami Original" (spice/variant marker one-sided)
    "herb",          # "Onion Garlic & Herb Dip" ≠ "Onion & Garlic Dip"
    "vegetable",     # "Vegetable Soup" ≠ "Creamy Tomato Soup"; "Vegetable Pilau" ≠ "Pilau"
    "crunchy",       # "Sweet & Crunchy Gherkins" ≠ plain "Sliced Gherkins"
    "lighter",       # "Flora Lighter Spread" ≠ "Flora Buttery Spread"
    "buttery",       # "Flora Buttery Spread" ≠ "Flora Lighter Spread"
    "protein",       # "Quaker Protein Gold" ≠ "Quaker Original" porridge
    "buttermilk",    # "Buttermilk Pancakes" ≠ plain "Pancakes"
    "creamy",        # "Creamy Tomato Soup" ≠ "Vegetable Soup"
    # Preparation / format discriminators
    "slimline",      # "Slimline Tonic Water" ≠ regular Tonic Water (lower calorie)
    "ground",        # "Ground Sweet Paprika" ≠ whole "Sweet Paprika"
    "croutons",      # "Soup with Croutons" ≠ plain Soup
    "skinless",      # "Skinless Salmon" ≠ regular Salmon fillet
    "boneless",      # "Boneless Chicken" ≠ regular chicken portions
    # Wine / champagne style discriminators
    "brut",          # "Brut Champagne" ≠ "Rosé Champagne"
    "vintage",       # "Vintage" wine ≠ non-vintage (NV) — distinct product
    # Dietary / formulation markers (asymmetric presence = different variant)
    "diet",          # "Diet Lemonade" ≠ "Lemonade"; "Diet Pepsi" ≠ "Pepsi"
    "milk",          # "Milk Chocolate" ≠ "Dark Chocolate"; "Aero Melts Caramel Milk" ≠ "Caramel"
    "caffeine",      # "Caffeine Free" ≠ regular — distinct product
    "peeled",        # "Peeled Brussels Sprouts" ≠ unpeeled (different preparation)
    # Product-line / range discriminators
    "nitro",         # "Funkin Nitro Cocktails" ≠ "Funkin Cocktails" (different format)
    "kids",          # "Innocent Kids" ≠ "Innocent" (different recipe/size range)
    "raw",           # "Raw Prawns" ≠ "Cooked Prawns" (different preparation state)
    "homestyle",     # "Homestyle Beef Gravy" ≠ plain "Beef Gravy" (recipe variant)
    "vitality",      # "MOJU Turmeric Vitality" ≠ "MOJU Turmeric Shots" (product line)
    "capsule",       # "Nespresso Capsules" ≠ loose coffee (different format)
    "capsules",      # plural form
    # Dry / style discriminators
    "dry",           # "Extra Dry Gin" ≠ plain "Gin"; "Dry Roasted" ≠ regular roasted
    # Ingredient-as-primary-descriptor discriminators
    "rice",          # "Chilli Con Carne & Rice" ≠ "Chilli Con Carne" (different dish)
    "broth",         # "Chicken Broth Soup" ≠ "Cream of Chicken Soup" (different soup)
    "cheese",        # "Creamy Cheese Pasta" ≠ "Creamy Pasta"; "Cheese Pizza" ≠ "Margherita"
    "pectin",        # "Jam Sugar with Added Pectin" ≠ plain "Jam Sugar"
    # Format / presentation discriminators
    "sandwich",      # "Ice Cream Sandwich" ≠ "Ice Cream"; "Sandwich Filler" ≠ "Sandwich"
    "filler",        # "Chicken Sandwich Filler" ≠ "Chicken Sandwich" (ingredient vs prepared)
    "mini",          # "Mini Naan" ≠ full-size Naan; "Mini Sausages" ≠ regular sausages
    "nibbles",       # "Caramel Nibbles" ≠ "Freddo Caramel" (different format/shape)
    "paste",         # "Jerk Seasoning Paste" ≠ "Jerk Seasoning" (paste vs dry)
    "nectar",        # "Pink Guava Nectar" ≠ "Finest Pink Guava" (different juice grade)
    # Wine / spirits style / quality discriminators
    "vsop",          # "Three Barrels VSOP Brandy" ≠ "Three Barrels Brandy" (quality grade)
    "pink",          # "Pink Moscato" ≠ "Moscato"; "Pink Gin" ≠ plain Gin (different style)
    # Dietary / fat-content discriminators
    "skinny",        # "Oatly Skinny" ≠ "Oatly Semi" (different fat content)
    "caramelised",   # "Caramelised Beetroot" ≠ plain "Beetroot" (different preparation)
    # Cooking / preparation state discriminators
    "battered",      # "Battered Cod Fillets" ≠ plain "Cod Fillets" (coating = different product)
    "fermented",     # "Fermented Beetroot Juice" ≠ "100% Beetroot Juice" (different process)
    # Cut / portion discriminators
    "legs",          # "Duck Legs" ≠ whole duck; "Chicken Legs" ≠ whole chicken portions
    # Spice / heat variant
    "fiery",         # "Fiery Chilli Beanz" ≠ "Chilli Beanz" — hotter variant
    # Inclusion / format discriminators
    "inclusions",    # "Crunchie Inclusions Ultimate Egg" ≠ plain "Ultimate Egg"
    "stuffed",       # "Stuffed Crust Pizza" ≠ standard "Crust Pizza"
    "pearl",         # "Sugar Pearl Waffles" ≠ "Sugar Waffles" (pearl = Belgian liège style)
    # Flavour-type discriminators not yet in FLAVOR_NAMED_TOKENS
    "soy",           # "Sweet Soy & Sea Salt Seaweed" ≠ "Sea Salt Seaweed" (different flavour)
    # Sweet as a primary flavour modifier — asymmetric = genuinely different recipe
    "sweet",         # "Sweet & Salted Popcorn" ≠ "Salted Popcorn"; "Sweet Pickle" ≠ "Pickle"
    # Spirits / wine quality discriminators — different grades are different products
    "anejo",         # "Havana Club Anejo Especial" ≠ "Havana Club Especial" (aged classification)
    "reserva",       # "Errazuriz Reserva Pinot Noir" ≠ "Errazuriz Estate Pinot Noir" (wine grade)
    # Certification / dietary production markers
    "halal",         # "Royal Halal Spaghetti Bolognese" ≠ "Royal Spaghetti Bolognese" (different production)
    # Product-line / variant markers
    "xtra",          # "Irn-Bru Xtra Sugar Free" ≠ "Irn-Bru Sugar Free" (reformulated, lower calorie)
    # Texture / finish discriminators
    "rough",         # "Rough Oatcakes" ≠ "Original Oatcakes" (different texture)
    # Cream as recipe discriminator — "Cream of Chicken" ≠ "Chicken Noodle" (different recipe)
    "cream",         # "Cream of Chicken Soup" ≠ "Chicken Noodle Soup"; "Cream Crackers" ≠ "Oatcakes"
    # Half/portion vs whole produce
    "half",          # "Half Cucumber" ≠ "Cucumber" (portion size = different SKU)
    # Noodle variety
    "udon",          # "Udon Noodles" ≠ generic "Straight to Wok Noodles"
    # Bread style
    "toastie",       # "White Toastie Bread" ≠ "Thickest White Bread" (different bread style)
    # Wine/spirits quality tier beyond "reserva" / "vsop"
    "private",       # "Private Reserve Malbec" ≠ "Reserve Malbec" (higher quality tier)
    # Raw dough vs finished product
    "dough",         # "Croissant Dough" ≠ baked "Croissants" (raw vs finished)
    # Dietary / nutrition product-line marker
    "carb",          # "Grenade Carb Killa" ≠ "Grenade Dark Chocolate" (protein bar line)
    # Ageing / coastal cask style (whisky)
    "sea",           # "Sea Cask" ≠ "Land Cask" whisky (different ageing environment)
    # Plant-based / vegan line marker
    "plant",         # "Cadbury Plant Chocolate" ≠ "Cadbury Dairy Milk" (dairy-free vs regular)
    # Produce portion discriminator
    "shredded",      # "Shredded Iceberg Lettuce" ≠ whole "Iceberg Lettuce" (different form)
    # Gold / golden — distinct flavour/product variant (Golden Roasted ≠ Original Roasted)
    "golden",        # "Whole Earth Drizzler Golden Roasted" ≠ "Original Roasted" peanut butter
    # Dish/format discriminators for formatted ready-meals
    "mash",          # "Liver & Bacon With Mash" ≠ plain "Liver & Bacon" (different dish)
    # Regional cooking style — Southern Style is a distinct marinade/preparation
    "southern",      # "Southern Style Chicken" ≠ "Slow Roasted Chicken" (different recipe)
    # Confectionery format discriminator — Bubbles format ≠ solid bar
    "bubbles",       # "Aero Bubbles" ≠ "Aero" bar (different confectionery format)
    # Wine product-line discriminator — Stamp ≠ Varietal Range within same brand
    "varietal",      # "Hardys Varietal Range Sauvignon Blanc" ≠ "Hardys Stamp Sauvignon Blanc"
    # Drink-type format discriminator — "Drinking Yoghurt" ≠ regular yoghurt pot
    "drinking",      # "Petits Filous Drinking Yoghurt" ≠ "Petits Filous" standard pot
    # Noodle / pasta width discriminator — width is a core product specification
    "wide",          # "Blue Dragon Wide Noodles" ≠ "Blue Dragon Medium Noodles"
    # Heat level — asymmetric presence (one-sided) = different heat variant
    "mild",          # "Nando's Peri BBQ Mild" ≠ "Nando's Peri BBQ" (lower heat = distinct product)
    # Texture variant
    "crispy",        # "M&M's Crispy Chocolate" ≠ "M&M's Chocolate" (different confectionery)
    # Certification / production method markers
    "fairtrade",     # "Fairtrade Shiraz" ≠ non-Fairtrade (certified supply chain = distinct product)
    # Regional style / marinade variant
    "chinese",       # "Chinese Style Pork Steaks" ≠ plain "Pork Steaks" (marinated variant)
    # Limited edition / heritage series discriminators
    "heritage",      # "Heritage Blend Rum" ≠ "Golden Rum" (special-edition expression)
    "gold",          # "Gold Caramel Billionaire Magnum" ≠ standard variant; "Nescafé Gold" ≠ Original
    # Size grade / noodle width
    "fine",          # "Fine Egg Noodles" ≠ "Medium Egg Noodles"; "Fine Sea Salt" ≠ regular salt
    # Colour as primary product differentiator
    "green",         # "Green Apple Sourz" ≠ "Apple Sourz"; "Green Tea" ≠ generic tea
    # Whisky blending / coffee production discriminator
    "blend",         # "Blended Scotch Whisky" ≠ "Single Malt"; "Coffee Blend" ≠ "Single Origin"
    # Colour as primary product differentiator (continued)
    "blood",         # "Blood Orange" ≠ plain "Orange" — distinct citrus variety
    # Beer style discriminator
    "session",       # "Hobgoblin Session IPA" ≠ "Hobgoblin IPA" (lower ABV, different product)
    # Vegetable as primary descriptor
    "squash",        # "Butternut Squash Soup" ≠ "Spinach & Cumin Soup" (different vegetable)
    # Coating / breading discriminator
    "breadcrumbs",   # "In Breadcrumbs" ≠ plain "Grills" / "Fillets" (different coating)
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
# Normalisation often strips these dietary / certification claims entirely, so
# token-level ONE_SIDED checks cannot see them; raw-name phrase checks are needed.
_NAS_RE           = re.compile(r'\bno\s+added\s+sugar\b', re.I)
_SUGAR_FREE_RE    = re.compile(r'\bsugar[\s\-]free\b', re.I)
_ORGANIC_RE       = re.compile(r'\borganic\b', re.I)
_VEGAN_RE         = re.compile(r'\bvegan\b', re.I)
_GLUTEN_FREE_RE   = re.compile(r'\bgluten[\s\-]free\b', re.I)
_DAIRY_FREE_RE    = re.compile(r'\bdairy[\s\-]free\b', re.I)
_PLANT_BASED_RE   = re.compile(r'\bplant[\s\-]based\b', re.I)
_ALCOHOL_FREE_RE  = re.compile(r'\balcohol[\s\-]free\b', re.I)
_LESS_SALT_RE     = re.compile(r'\bless\s+salt\b|\breduced\s+salt\b|\blow\s+salt\b', re.I)
_LOW_SUGAR_RE     = re.compile(r'\blow[\s\-]sugar\b', re.I)
_LOW_FAT_RE       = re.compile(r'\blow[\s\-]fat\b', re.I)
_FAT_FREE_RE      = re.compile(r'\bfat[\s\-]free\b', re.I)
_REDUCED_FAT_RE   = re.compile(r'\breduced[\s\-]fat\b', re.I)
_FREE_RANGE_RE    = re.compile(r'\bfree[\s\-]range\b', re.I)
_VEGETARIAN_RE    = re.compile(r'\bvegetarian\b', re.I)
_LESS_SUGAR_RE    = re.compile(r'\bless\s+sugar\b', re.I)
_LESS_FAT_RE      = re.compile(r'\bless\s+fat\b', re.I)


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
    cannot be expressed as individual token lookups — normalisation strips these
    dietary and certification claims before token checks run.

    Asymmetry in any of the patterns below constitutes a hard conflict:
      - "No Added Sugar" / "Sugar Free" — different dietary formulations
      - "Organic" — regulated certification; organic vs non-organic is a different SKU
      - "Vegan" — recipe-level reformulation (e.g. Hellmann's Vegan ≠ Real Mayo)
      - "Gluten Free" / "Dairy Free" / "Plant Based" — free-from variants
      - "Alcohol Free" — e.g. Kopparberg Alcohol Free ≠ regular Kopparberg
      - "Less Salt" / "Reduced Salt" / "Low Salt" — reduced-sodium variant
      - "Low Sugar" — e.g. Volvic Touch of Fruit Low Sugar ≠ regular
      - "Low Fat" — e.g. Activia Low Fat ≠ full-fat variant
      - "Vegetarian" — e.g. Simon Howie Vegetarian Haggis ≠ Original Haggis
    """
    for regex in (
        _NAS_RE,
        _SUGAR_FREE_RE,
        _ORGANIC_RE,
        _VEGAN_RE,
        _GLUTEN_FREE_RE,
        _DAIRY_FREE_RE,
        _PLANT_BASED_RE,
        _ALCOHOL_FREE_RE,
        _LESS_SALT_RE,
        _LOW_SUGAR_RE,
        _LOW_FAT_RE,
        _FAT_FREE_RE,
        _REDUCED_FAT_RE,
        _FREE_RANGE_RE,
        _VEGETARIAN_RE,
        _LESS_SUGAR_RE,
        _LESS_FAT_RE,
    ):
        has_a = bool(regex.search(raw_name_a))
        has_b = bool(regex.search(raw_name_b))
        if has_a != has_b:
            return True

    return False
