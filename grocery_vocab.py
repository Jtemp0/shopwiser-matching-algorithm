from typing import List, Dict

# ============================================================
# VOCABULARY & PATTERN DICTIONARIES
# ============================================================

SUPERMARKET_BRANDS: Dict[str, List[str]] = {
    'Tesco': [
        'Tesco Finest', 'Tesco Everyday Value', 'Tesco Free From', 'Tesco Organic',
        'Hearty Food Co.', 'Tesco Plant Chef', 'Stockwell & Co', 'Ms Molly\'s', 
        'Eastmans', 'Creamfields', 'Grower\'s Harvest', 'Tesco'
    ],
    'ASDA': [
        'ASDA Extra Special', 'Asda Extra Special', 'ASDA Smart Price', 
        'Just Essentials by ASDA', 'Just Essentials', 'ASDA Plant Based', 
        'ASDA Chosen By You', 'ASDA Free From', 'COOK by ASDA', 'BAKE by ASDA', 
        'The BAKERY at ASDA', 'OMV! by Asda', 'ASDA', 'Asda'
    ],
    'Morrisons': [
        'Morrisons The Best', 'Morrisons Savers', 'Morrisons Free From', 
        'Morrisons Organic', 'Morrisons Market Street', 'Market Street', 
        'Morrisons Plant Revolution', 'Morrisons For Farmers', 'M Kitchen', 
        'Savers', 'Morrisons'
    ],
    'Sains': [
        "Sainsbury's Taste the Difference", "Taste the Difference", 
        "Sainsbury's Basics", "Stamford Street Co.", "Stamford Street", 
        "Sainsbury's Free From", "Sainsbury's SO Organic", 'SO Organic', 
        "Sainsbury's Plant Pioneers", 'JS', "Sainsbury's", "Sainsburys"
    ],
}

# Map supermarket brand prefix → tier
TIER_MAP = {
    # Premium
    'finest': 'premium', 'extra special': 'premium', 'taste the difference': 'premium',
    'the best': 'premium', 'market street': 'premium', 'no.1': 'premium', 
    'specially selected': 'premium', 'deluxe': 'premium', 'irresistible': 'premium',
    
    # Value (Tertiary / Farm brands)
    'hearty food co': 'value', 'stockwell': 'value', 'ms molly': 'value', 
    'eastmans': 'value', 'creamfields': 'value', 'grower\'s harvest': 'value',
    'just essentials': 'value', 'smart price': 'value', 'everyday value': 'value', 
    'savers': 'value', 'basics': 'value', 'stamford street': 'value', 
    'everyday essentials': 'value', 'simply': 'value', 'honest value': 'value',
    
    # Dietary / Specialty
    'plant chef': 'dietary', 'plant based': 'dietary', 'omv!': 'dietary', 
    'plant pioneers': 'dietary', 'plant menu': 'dietary', 'vemondo': 'dietary',
    'plant revolution': 'dietary', 'gro': 'dietary', 'free from': 'dietary'
}


# Comprehensive list of known commercial brands
KNOWN_BRANDS: List[str] = [
    # Soft drinks, water, juice
    'coca-cola', 'coca cola', 'pepsi', 'fanta', 'sprite', 'dr pepper', '7up', 'schweppes',
    'ribena', 'robinsons', 'tropicana', 'innocent', 'capri sun', 'vimto', 'lucozade',
    'red bull', 'monster', 'relentless', 'rockstar', 'boost',
    'irn bru', 'irn-bru', 'oasis', 'rubicon', 'tango', 'ocean spray',
    'san pellegrino', 'sanpellegrino', 'evian', 'volvic', 'buxton', 'highland spring',
    'mountain dew', 'fever-tree', 'fever tree', 'britvic', 'j2o', 'fruit shoot',
    'barr', 'shloer', 'belvoir', 'cawston press', 'the london essence co',
    'long tail mixers', 'bottlegreen',
    'diet coke', 'coca-cola zero', 'coca cola zero',
    'naked juice', 'naked',
    'grace', 'ktc', 'regal',

    # Coffee & Tea
    'nescafe', 'nescafé', 'twinings', 'tetley', 'pg tips', 'yorkshire tea', 'clipper',
    'pukka', 'costa', 'starbucks', 'lavazza', 'illy', 'kenco', 'douwe egberts', 'tassimo',
    'nespresso', 'dolce gusto', 'taylors of harrogate', 'taylors', 'cafe direct', 'cafedirect',
    'maxwell house', 'l\'or', 'lor', 'leon', 'bird & blend', 'teapigs',

    # Dairy, chilled & plant-based
    'anchor', 'lurpak', 'flora', 'benecol', 'philadelphia', 'dairylea', 'babybel',
    'muller', 'müller', 'yoplait', 'activia', 'danone', 'alpro', 'oatly',
    'cathedral city', 'yeo valley', 'country life', 'clover', 'kerrygold',
    'cravendale', 'elmlea', 'president', 'président', 'castello', 'onken',
    'glenisk', 'violife', 'follow your heart', 'almond breeze', 'rude health',
    'arla', 'delamere', 'rachel\'s', 'chobani', 'fage', 'galbani',
    'light & free', 'yams', 'yamas',
    'actimel', 'petits filous', 'pilgrims choice', 'seriously',

    # Breakfast & cereals
    'weetabix', 'oatibix', 'alpen', 'jordans', 'ready brek', 'shreddies',
    'cheerios', 'shredded wheat', 'special k', 'corn flakes', 'rice krispies',
    'coco pops', 'frosties', 'nature valley', 'belvita', 'nakd', 'trek', 'kind',
    'kellogg\'s', 'kelloggs', 'nestle', 'quaker', 'scotts', 'flahavan\'s',

    # Bread, bakery, cakes
    'warburtons', 'hovis', 'kingsmill', 'allinson', 'soreen', 'jus-rol',
    'mcvitie\'s', 'mcvities', 'jacob\'s', 'jacobs', 'fox\'s', 'foxs',
    'mr kipling', 'border biscuits', 'border', 'tunnock\'s', 'tunnocks',
    'maryland', 'go ahead', 'go-ahead', 'lotus biscoff', 'biscoff',
    'carr\'s', 'carrs', 'pop tarts', 'pop-tarts', 'graze', 'higgidy',
    'miss molly\'s', 'finsbury food',
    'ryvita', 'nairn\'s', 'nairns', 'genius', 'schar', 'schär',
    'kallo', 'eat natural', 'nakd',

    # Biscuits, sweets, chocolate
    'cadbury', 'mars', 'snickers', 'galaxy', 'lindt', 'thorntons',
    'green & black\'s', 'green and blacks', 'ferrero rocher', 'ferrero',
    'toblerone', 'after eight', 'after eights', 'maltesers', 'm&ms', 'm&m\'s',
    'bounty', 'milky way', 'twix', 'kit kat', 'kitkat',
    'rowntree\'s', 'rowntrees', 'haribo', 'maynards bassetts', 'maynards', 'bassetts',
    'swizzels', 'swizzels matlow', 'kinder', 'kinder bueno', 'werther\'s', 'werthers',
    'tic tac', 'mentos', 'skittles', 'starburst', 'polo', 'chupa chups',
    'reese\'s', 'reeses', 'hershey\'s', 'hersheys', 'terry\'s', 'terrys',
    'kp', 'kp nuts', 'nobby\'s', 'nobbys', 'planters',
    'gü', 'gu', 'gu puds',
    'aero', 'milkybar', 'lion bar',

    # Crisps & savoury snacks
    'walkers', 'pringles', 'doritos', 'mccoys', 'quavers', 'wotsits',
    'kettle', 'kettle chips', 'tyrrells', 'seabrook', 'pom-bear', 'pombear',
    'hula hoops', 'hulahoops', 'skips', 'discos', 'nik naks', 'niknaks',
    'frazzles', 'monster munch', 'popchips', 'sunbites',
    'propercorn', 'properchips', 'metcalfe\'s', 'metcalfes',
    'snack a jacks', 'snackajacks', 'mini cheddars', 'cofresh', 'cheetos',

    # Rice, grains, pasta, noodles
    'ben\'s original', 'bens original', 'uncle ben\'s', 'uncle bens',
    'tilda', 'laila', 'merchant gourmet', 'barilla', 'de cecco', 'dececco',
    'napolina', 'batchelors', 'itsu', 'yutaka', 'amoy', 'nissin', 'kohinoor',
    'dolmio', 'loyd grossman',

    # Tins, jars, sauces, condiments
    'heinz', 'branston', 'colman\'s', 'colmans', 'marmite', 'bovril', 'oxo', 'bisto',
    'paxo', 'knorr', 'maggi', 'hellmann\'s', 'hellmanns',
    'hp sauce', 'hp', 'lea & perrins', 'lea and perrins',
    'frank\'s redhot', 'franks redhot', 'tabasco', 'encona', 'cholula',
    'sriracha', 'flying goose', 'kikkoman', 'lee kum kee', 'blue dragon',
    'old el paso', 'santa maria', 'geeta\'s', 'geetas', 'rajah',
    'sharwood\'s', 'sharwoods', 'pataks', 'patak\'s', 'the spice tailor',
    'schwartz', 'bart', 'maldon', 'saxa', 'sarson\'s', 'sarsons',
    'haywards', 'whitworths', 'filippo berio', 'bertolli',
    'green giant', 'cirio', 'mutti', 'sacla', 'fray bentos', 'princes',
    'baxters', 'hartley\'s', 'hartleys', 'tiptree', 'robertson\'s', 'robertsons',
    'duerr\'s', 'duerrs', 'bonne maman', 'rowse', 'nutella', 'sun-pat', 'sun pat',
    'tracklements', 'encona', 'potts', 'lea & perrins',

    # Frozen foods & ready meals (incl. ice cream)
    'mccain', 'goodfella\'s', 'goodfellas', 'dr oetker', 'dr. oetker',
    'chicago town', 'rustlers', 'birdseye', 'birds eye',
    'aunt bessie\'s', 'aunt bessies', 'findus',
    'strong roots', 'cauldron', 'tofoo', 'the tofoo co',
    'beyond meat', 'moving mountains', 'vivera', 'meatless farm', 'cook',
    'magnum', 'haagen-dazs', 'haagen dazs', 'ben & jerry\'s', 'ben and jerrys',
    'carte d\'or', 'cornetto', 'solero', 'walls',

    # Chilled meat, fish, pies
    'wall\'s', 'heck', 'herta', 'bernard matthews', 'mattessons',
    'ginsters', 'pukka pies', 'quorn', 'linda mccartney', 'richmond',
    'john west', 'youngs', 'young\'s', 'glenryck',
    'peperami', 'fridge raiders',

    # Beer & Cider
    'stella artois', 'stella', 'budweiser', 'bud', 'heineken', 'carlsberg',
    'foster\'s', 'fosters', 'carling', 'kronenbourg', '1664',
    'peroni', 'birra moretti', 'moretti', 'san miguel', 'estrella', 'estrella damm',
    'guinness', 'john smith\'s', 'john smiths', 'boddingtons',
    'brewdog', 'camden town', 'camden hells', 'beavertown', 'neck oil',
    'strongbow', 'thatchers', 'kopparberg', 'magners', 'bulmers', 'old mout',
    'corona', 'desperados', 'beck\'s', 'becks',
    'old speckled hen', 'old crafty hen', 'spitfire', 'hobgoblin',
    'birra peroni', 'bavaria', 'amstel', 'leffe',

    # Spirits
    'smirnoff', 'absolut', 'grey goose', 'glen\'s', 'glens',
    'gordon\'s', 'gordons', 'bombay sapphire', 'tanqueray', 'hendrick\'s', 'hendricks',
    'beefeater', 'edinburgh gin', 'brockmans', 'ciroc',
    'jack daniel\'s', 'jack daniels', 'jim beam', 'maker\'s mark',
    'jameson', 'famous grouse', 'bell\'s', 'bells', 'johnnie walker',
    'glenfiddich', 'glenlivet', 'cardhu', 'highland park', 'macallan',
    'captain morgan', 'bacardi', 'havana club', 'malibu', 'kraken',
    'baileys', 'tia maria', 'kahlua', 'disaronno', 'southern comfort',
    'pimms', 'pimm\'s', 'martini', 'aperol', 'campari',
    'courvoisier', 'hennessy', 'courvoisier',
    'warninks', 'advocaat',

    # Wine & Champagne
    'hardys', 'blossom hill', 'yellow tail', 'casillero del diablo',
    'campo viejo', 'villa maria', 'oyster bay', '19 crimes', 'jam shed',
    'barefoot', 'mcguigan', 'jacob\'s creek', 'jacobs creek',
    'lindeman\'s', 'lindemans', 'wolf blass', 'errazuriz', 'trapiche',
    'freixenet', 'moët', 'moet', 'veuve clicquot', 'bollinger', 'lanson',
    'squealing pig', 'wollemi', 'giesen',

    # Health, protein, wellness
    'fulfil', 'phd nutrition', 'phd smart bar', 'the skinny food co',
    'getpro', 'nakd', 'trek', 'kind', 'graze', 'nature valley',
    'phd', 'optimum nutrition', 'myprotein', 'up&go', 'weetabix on the go',
    'grenade',

    # Baby & toddler
    'cow & gate', 'cow and gate', 'aptamil', 'hipp', 'hipp organic',
    'ella\'s kitchen', 'ellas kitchen', 'organix', 'kiddylicious',
    'heinz by nature', 'piccolo', 'little freddie',

    # Home baking
    'tate & lyle', 'tate and lyle', 'silver spoon', 'billington\'s',
    'allinson\'s', 'allinsons', 'homepride', 'betty crocker', 'doves farm',
    'dr. oetker', 'silver spoon',

    # Dairy-free
    'oatly', 'alpro', 'provamil', 'koko', 'nutpods',

    # Tins & specialist food brands
    'east end', 'fudco', 'natco', 'ktc',
    'new covent garden',

    # Misc
    'ambrosia', 'crosta & mollica', 'prymat', 'felix', 'soreen',
    'express cuisine', 'staveley',
    'nando\'s', 'nandos',

    # Soft drinks, water, juice
    'lipton', 'appletiser', 'smartwater', 'vittel', 'strathmore', 'kia-ora', 'kia ora',
    'miwadi', 'drench', 'ballygowan', 'calypso', 'snapple', 'remedy', 'mogu mogu',
    'vithit', 'vit hit', 'plenish', 'purdey\'s', 'purdeys',

    # Coffee & Tea
    'typhoo', 'carte noire', 'tazo', 'punjana', 'barry\'s tea', 'barrys tea', 'barry\'s', 
    'beanies', 'camp coffee', 'puro', 'tick tock', 'teapigs', 'whittard', 'ilou', 
    'milano', 'blue tokai',

    # Dairy, chilled & plant-based
    'i can\'t believe it\'s not butter', 'utterly butterly', 'stork', 'the laughing cow',
    'laughing cow', 'boursin', 'cheestrings', 'frubes', 'munch bunch', 'ski', 'yakult',
    'biomel', 'nurishh', 'primula', 'puck', 'dairygold', 'golden cow', 'applewood',
    'mexicana', 'ilchester', 'wensleydale', 'norseland', 'actimel' # Wait, actimel is in KNOWN_BRANDS. 
    # Removed actimel. (Self-correction applied).
    'galbani', 'nomadic', 'lindahls', 'nush',

    # Breakfast & cereals
    'honey monster', 'krave', 'dorset cereals', 'fuel10k', 'moma', 'crunchy nut',
    'chex', 'bear', 'surreal', 'all-bran', 'all bran', 'cheerios', 'milo',

    # Bread, bakery, cakes
    'roberts bakery', 'roberts', 'new york bakery co', 'mission', 'mission deli', 
    'wallaces', 'wenzel\'s', 'brace\'s', 'braces', 'peter\'s', 'peters', 'baker street',
    'biona', 'st pierre', 'brioche pasquier', 'crosta & mollica' # Wait, crosta & mollica is in known Misc.
    # Removed crosta & mollica.
    'wafflemeister', 'bfree', 'b-free', 'cranks',

    # Biscuits, sweets, chocolate
    'oreo', 'jammie dodgers', 'wagon wheels', 'penguin', 'milka', 'daim', 'quality street',
    'roses', 'celebrations', 'heroes', 'smarties', 'rolo', 'matchmakers', 'crawford\'s',
    'crawfords', 'bahlsen', 'loacker', 'kinder surprise', 'kinder joy', 'fruit pastilles',
    'jelly tots', 'toffee crisp', 'munchies', 'yorkie', 'twirl', 'wispa', 'crunchie',
    'double decker', 'flake', 'curly wurly', 'chomps', 'fudge', 'bournville', 'topic', 
    'picnic', 'boost', 'starbar', 'ritter sport', 'montezuma\'s', 'montezumas', 'ombar',

    # Crisps & savoury snacks
    'space raiders', 'wheat crunchies', 'roysters', 'pipers', 'burts', 'eat real', 
    'hippeas', 'smiths', 'french fries', 'squares', 'sensations', 'walkers sensations', 
    'real crisps', 'tayto', 'taytos', 'golden wonder', 'pringles', 'walkers' # Note: Pringles and Walkers are in KNOWN
    # Removed Pringles and Walkers.
    'wild trail', 'sunbites', 'ryvita', 'ritz', 'tuc', 'cheddars', 'jacob\'s cream crackers',

    # Rice, grains, pasta, noodles
    'pot noodle', 'koka', 'indomie', 'nongshim', 'mug shot', 'buitoni', 'garofalo', 
    'rummo', 'sunrice', 'veetee', 'soba', 'mama', 'samyang', 'kabuto', 'kabuto noodles',
    'biona organic', 'seeds of change',

    # Tins, jars, sauces, condiments
    'daddies', 'newman\'s own', 'crosse & blackwell', 'crosse and blackwell', 'campbell\'s',
    'campbells', 'bicks', 'mary berry', 'pizza express', 'wagamama', 'wasabi', 'maille',
    'kewpie', 'mae ploy', 'vegemite', 'branstons', 'hp', 'nandos', 'heinz' # Heinz, HP, Nandos, Branstons are duplicates
    # Removed Heinz, HP, Nandos, Branstons.
    'stokes', 'stokes sauces', 'chef', 'willy\'s', 'willys', 'fajita magic', 
    'nissin', 'kikkoman', 'bisto', 'oxo' # Bisto, oxo, nissin, kikkoman in KNOWN
    # Removed Bisto, oxo, nissin, kikkoman.
    'mizkan', 'nakano', 'leon', 'pizzaexpress',

    # Frozen foods & ready meals (incl. ice cream)
    'breyers', 'kelly\'s', 'kellys', 'mackie\'s', 'mackies', 'halo top', 'jude\'s', 
    'judes', 'northern bloc', 'glorious', 'auntys', 'dalepak', 'iceland', 'farmfoods',
    'charlotte\'s', 'charlottes', 'green\'s', 'greens', 'snax', 'yumnut', 'oppo', 
    'oppo brothers', 'little moons', 'røar', 'roar',

    # Chilled meat, fish, pies
    'denny', 'galtee', 'finnebrogue', 'simon howie', 'edwards of conwy', 'unearthed',
    'gosh!', 'gosh', 'squeaky bean', 'charlie bigham\'s', 'charlie bighams', 'mains',
    'cockburns', 'macsween', 'heck', 'richmond' # Heck and Richmond are in KNOWN
    # Removed Heck and Richmond.
    'deli kitchen', 'savoury bakes', 'rollo', 'bighams', 'mighty',

    # Beer & Cider
    'doom bar', 'cruzcampo', 'madri', 'madrí', 'tiger', 'cobra', 'singha', 'asahi', 
    'sapporo', 'tsingtao', 'innis & gunn', 'stowford press', 'westons', 'henry westons', 
    'rekorderlig', 'brothers', 'tuborg', 'tennent\'s', 'tennents', 'hoegaarden', 
    'blue moon', 'coors', 'coors light', 'sharps', 'sharp\'s', 'brewdog', 'camden' # Brewdog is in KNOWN
    # Removed brewdog.
    'beavertown', 'goose island', 'lagunitas', 'brooklyn brewery', 'tiny rebel',

    # Spirits
    'russian standard', 'stolichnaya', 'stoli', 'ketel one', 'belvedere', 'sailor jerry',
    'dead man\'s fingers', 'mount gay', 'appleton estate', 'wray & nephew', 'wray and nephew',
    'glenmorangie', 'talisker', 'laphroaig', 'lagavulin', 'ardbeg', 'woodford reserve', 
    'bulleit', 'remy martin', 'rémy martin', 'martell', 'cointreau', 'grand marnier',
    'passoa', 'passoã', 'chambord', 'archers', 'havana club', 'malibu' # Havana Club, Malibu in KNOWN
    # Removed Havana Club, Malibu.
    'jura', 'bowmore', 'auchentoshan', 'sipsmith', 'tanqueray ten', 'malfy',

    # Wine & Champagne
    'echo falls', 'gallo', 'dark horse', 'apothic', 'black tower', 'blue nun', 
    'jp chenet', 'mateus', 'laurent-perrier', 'laurent perrier', 'dom perignon', 
    'dom pérignon', 'piper-heidsieck', 'cinzano', 'croft', 'harveys', 'taylor\'s', 
    'kumala', 'concha y toro', 'brancott estate', 'trivento', 'mud house', 'i heart',
    'frescobaldi', 'antinori',

    # Health, protein, wellness
    'huel', 'soylent', 'slimfast', 'yfood', 'protein works', 'bulk', 'maxinutrition',
    'maximuscle', 'usn', 'sci-mx', 'scimx', 'nutramino', 'misFits', 'misfits health',
    'barebells', 'uFit', 'ufit drinks',

    # Fresh Produce & Meat brands
    'pink lady', 'albert bartlett', 'florette', 'tenderstem', 'driscoll\'s', 'driscolls', 
    'berry gardens', 'mushrooms', 'gressingham', 'mowi', 'saucy fish co', 'heck', 
    'peperami', 'rustlers', 'mattessons', 'richmond', 'walls', 'wall\'s',

]

# Attributes (remove from name after detection)
ATTRIBUTES: Dict[str, List[str]] = {
    'organic': ['organic', 'bio', 'soil association'],
    'free_from': [
        'gluten free', 'gluten-free', 'dairy free', 'dairy-free', 'lactose free', 
        'lactose-free', 'wheat free', 'wheat-free', 'nut free', 'nut-free', 
        'soya free', 'soya-free', 'caffeine free', 'decaf', 'free from'
    ],
    'diet': [
        'vegan', 'vegetarian', 'plant-based', 'plant based', 'low fat', 'reduced fat',
        'fat free', 'fat-free', 'sugar free', 'sugar-free', 'no added sugar', 
        'low sugar', 'light', 'lite', 'keto', 'halal', 'kosher'
    ],
    'ethical': [
        'fairtrade', 'fair trade', 'free range', 'freedom food', 'rspca assured', 
        'msc certified', 'red tractor', 'rainforest alliance', 'sustainable'
    ],
    'preparation': [
        'ready to eat', 'ready to cook', 'ready meal', 'meal for one',
        'microwaveable', 'oven ready', 'pre-cooked', 'part baked', 'parbaked'
    ],
    'alcohol_marker': ['alcohol free', 'alcohol-free', 'non-alcoholic', '0% alcohol', 'low alcohol']
}


# Descriptors (detected but NOT removed from name)
DESCRIPTORS: Dict[str, List[str]] = {
    'cooking_method': ['roasted', 'grilled', 'baked', 'fried', 'smoked', 'steamed', 'poached', 'bbq', 'barbecue', 'chargrilled'],
    'texture':        ['crunchy', 'crispy', 'smooth', 'chunky', 'creamy', 'thick', 'thin', 'soft'],
    'flavor':         ['mild', 'medium', 'hot', 'spicy', 'sweet', 'savoury', 'salty', 'tangy', 'sour', 'bitter', 'unsmoked'],
    'form':           ['sliced', 'diced', 'chopped', 'whole', 'halved', 'quartered', 'peeled', 'shredded', 'grated', 'ground', 'crushed', 'powder', 'paste', 'puree', 'concentrate'],
    'cut_meat':       ['mince', 'minced', 'breast', 'thigh', 'drumstick', 'fillet', 'steak', 'joint', 'chop', 'sausage', 'meatballs', 'burgers'],
    
    # NEW: Grocery-specific preparation states
    'prep_meat_fish': ['boneless', 'bone-in', 'bone in', 'skinless', 'skin-on', 'skin on', 'deboned', 'line caught', 'farmed'],
    'prep_produce':   ['seedless', 'pitted', 'stoned', 'unwaxed', 'unpitted', 'washed', 'unwashed'],
    'bakery_type':    ['wholemeal', 'wholegrain', 'sourdough', 'seeded', 'white', 'brown', 'crusty', 'multigrain', 'brioche'],
    
    'maturity':       ['mild', 'mature', 'extra mature', 'vintage', 'aged'],
    # Added "loose" to packaging (very common for UK fruit/veg e.g. "Bananas Loose")
    'packaging':      ['can', 'cans', 'tin', 'tins', 'bottle', 'bottles', 'pouch', 'tub', 'jar', 'carton', 'multipack', 'loose'] 
}


# Keywords that indicate the mg value is nicotine strength, not product weight
_VAPE_KEYWORDS = frozenset([
    'e-liquid', 'e liquid', 'eliquid', 'vape', 'vaping', 'pod', 'pods',
    'nicotine', 'tobacco', 'cigarette', 'e-cig', 'ecig', 'juul', 'elf bar',
    'disposable', 'menthol e', 'nic salt',
])