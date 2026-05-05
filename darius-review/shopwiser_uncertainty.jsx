import { useState, useMemo, useEffect } from "react";
import {
  Search, X, Plus, Minus, ShieldCheck, ShieldAlert, Shield,
  AlertTriangle, Info, Filter, ArrowRight, ChevronDown, ChevronUp,
} from "lucide-react";

// === Demo data: 62 clusters spanning the full confidence range ===
const DEMO_DATA = [{"id":7443,"name":"Bonne Maman Lemon Curd 360g","category":"food_cupboard","confidence":1.0,"issues":[],"items":[{"shop":"ASDA","price":2.6,"unit_price":7.2,"unit":"kg","title":"Bonne Maman Lemon Curd 360g","size_g":360.0},{"shop":"Morrisons","price":2.89,"unit_price":8.0,"unit":"kg","title":"Bonne Maman Lemon Curd","size_g":360.0},{"shop":"Sains","price":2.0,"unit_price":5.6,"unit":"kg","title":"Bonne Maman Lemon Curd 360g","size_g":360.0},{"shop":"Tesco","price":2.8,"unit_price":7.8,"unit":"kg","title":"Bonne Maman Lemon Curd 360g","size_g":360.0}]},{"id":2328,"name":"Sipsmith London Dry Gin 700ml","category":"drinks","confidence":1.0,"issues":[],"items":[{"shop":"ASDA","price":30.0,"unit_price":42.86,"unit":"l","title":"Sipsmith London Dry Gin 700ml","size_g":700.0},{"shop":"Morrisons","price":30.0,"unit_price":42.86,"unit":"l","title":"Sipsmith London Dry Gin 41.6%","size_g":700.0},{"shop":"Sains","price":30.0,"unit_price":42.86,"unit":"l","title":"Sipsmith London Dry Gin 70cl","size_g":700.0},{"shop":"Tesco","price":30.0,"unit_price":42.86,"unit":"l","title":"SIPSMITH LONDON DRY GIN 70CL","size_g":700.0}]},{"id":9481,"name":"Pukka All Day Breakfast Slice 170g","category":"fresh_food","confidence":1.0,"issues":[],"items":[{"shop":"ASDA","price":1.75,"unit_price":10.3,"unit":"kg","title":"Pukka All Day Breakfast Slice 170g","size_g":170.0},{"shop":"Morrisons","price":1.25,"unit_price":7.4,"unit":"kg","title":"Pukka All Day Breakfast Slice","size_g":170.0},{"shop":"Sains","price":1.75,"unit_price":10.3,"unit":"kg","title":"Pukka All Day Breakfast Slice 170g","size_g":170.0},{"shop":"Tesco","price":1.75,"unit_price":10.3,"unit":"kg","title":"Pukka All Day Breakfast Slice 170g","size_g":170.0}]},{"id":3799,"name":"Weetabix On The Go Breakfast Drink Vanilla 250ml","category":"food_cupboard","confidence":1.0,"issues":[],"items":[{"shop":"ASDA","price":1.49,"unit_price":6.0,"unit":"l","title":"Weetabix On the Go Breakfast Drink Vanilla","size_g":250.0},{"shop":"Morrisons","price":1.5,"unit_price":6.0,"unit":"l","title":"Weetabix On the Go Breakfast Drink Vanilla","size_g":250.0},{"shop":"Sains","price":1.5,"unit_price":6.0,"unit":"l","title":"Weetabix On The Go Breakfast Drink Vanilla 250ml","size_g":250.0},{"shop":"Tesco","price":1.5,"unit_price":6.0,"unit":"l","title":"Weetabix On The Go Vanilla Drink 250Ml","size_g":250.0}]},{"id":4148,"name":"Heinz Lamb & Vegetable Chunky Big Soup 400g","category":"food_cupboard","confidence":1.0,"issues":[],"items":[{"shop":"ASDA","price":2.0,"unit_price":5.0,"unit":"kg","title":"Heinz Lamb & Vegetable Chunky Big Soup","size_g":400.0},{"shop":"Morrisons","price":2.0,"unit_price":5.0,"unit":"kg","title":"Heinz Lamb & Vegetable Chunky Big Soup","size_g":400.0},{"shop":"Sains","price":2.0,"unit_price":5.0,"unit":"kg","title":"Heinz Lamb & Vegetable Chunky Big Soup 400g","size_g":400.0},{"shop":"Tesco","price":2.0,"unit_price":5.0,"unit":"kg","title":"Heinz Big Soup Lamb And Vegetable 400G","size_g":400.0}]},{"id":3507,"name":"Aero Milk Chocolate Mousse 4x59g","category":"fresh_food","confidence":0.994,"issues":[],"items":[{"shop":"ASDA","price":1.5,"unit_price":6.4,"unit":"kg","title":"Aero Milk Chocolate Mousse","size_g":240.0},{"shop":"Morrisons","price":1.5,"unit_price":6.4,"unit":"kg","title":"Aero Chocolate Mousse","size_g":240.0},{"shop":"Sains","price":1.25,"unit_price":5.3,"unit":"kg","title":"Aero Milk Chocolate Mousse 4x59g","size_g":236.0},{"shop":"Tesco","price":1.25,"unit_price":5.3,"unit":"kg","title":"Aero Chocolate Mousse 4 X59g","size_g":236.0}]},{"id":12107,"name":"Kinder Bueno White Chocolate & Hazelnuts Bars Multipack 4x39g","category":"food_cupboard","confidence":0.988,"issues":[],"items":[{"shop":"ASDA","price":2.55,"unit_price":16.3,"unit":"kg","title":"Kinder Bueno Hazelnuts & White Chocolate Bars Multipack 4x","size_g":160.0},{"shop":"Morrisons","price":2.0,"unit_price":12.8,"unit":"kg","title":"Kinder Bueno White Chocolate & Hazelnuts Bars Multipack","size_g":160.0},{"shop":"Sains","price":2.25,"unit_price":14.4,"unit":"kg","title":"Kinder Bueno White Chocolate & Hazelnuts Bars Multipack 4x39g","size_g":156.0},{"shop":"Tesco","price":2.55,"unit_price":16.4,"unit":"kg","title":"Kinder Bueno White Chocolate Bars Multipack 4 X 39g","size_g":156.0}]},{"id":2457,"name":"Yellow Tail Shiraz Red Wine","category":"drinks","confidence":0.958,"issues":[],"items":[{"shop":"ASDA","price":7.0,"unit_price":93.31,"unit":"l","title":"Yellow Tail Shiraz Red Wine","size_g":750.0},{"shop":"Morrisons","price":7.0,"unit_price":9.31,"unit":"l","title":"Yellow Tail Shiraz","size_g":750.0},{"shop":"Sains","price":7.0,"unit_price":9.31,"unit":"l","title":"Yellow Tail Shiraz 75cl","size_g":750.0},{"shop":"Tesco","price":7.75,"unit_price":10.31,"unit":"l","title":"Yellow Tail Shiraz 75Cl","size_g":750.0}]},{"id":1630,"name":"Maryland Cookies Chocolate Chip 200g","category":"food_cupboard","confidence":0.958,"issues":[],"items":[{"shop":"ASDA","price":0.9,"unit_price":4.5,"unit":"kg","title":"Maryland Cookies Choc Chip","size_g":200.0},{"shop":"Morrisons","price":0.9,"unit_price":4.5,"unit":"kg","title":"Maryland Cookies Chocolate Chip","size_g":200.0},{"shop":"Sains","price":1.5,"unit_price":7.5,"unit":"kg","title":"Maryland Cookies Chocolate Chip 200g","size_g":200.0},{"shop":"Tesco","price":0.9,"unit_price":4.5,"unit":"kg","title":"Maryland Chocolate Chip Cookies 200G","size_g":200.0}]},{"id":5817,"name":"Warburtons Plain Thin Bagels x6","category":"free_from","confidence":0.95,"issues":["category_conflict"],"items":[{"shop":"ASDA","price":1.35,"unit_price":0.23,"unit":"unit","title":"Warburtons Thin Bagels Plain","size_g":null},{"shop":"Morrisons","price":1.69,"unit_price":0.28,"unit":"unit","title":"Warburtons Thin Plain Bagels","size_g":null},{"shop":"Sains","price":1.25,"unit_price":0.21,"unit":"unit","title":"Warburtons Plain Thin Bagels x6","size_g":null},{"shop":"Tesco","price":1.5,"unit_price":0.3,"unit":"unit","title":"Warburtons 5 Pack Bagels Plain","size_g":null}]},{"id":6402,"name":"Oatly Oat Drink Semi Chilled 1L","category":"drinks","confidence":0.95,"issues":["category_conflict"],"items":[{"shop":"ASDA","price":2.1,"unit_price":2.1,"unit":"l","title":"Oatly Oat Drink Semi","size_g":1000.0},{"shop":"Morrisons","price":2.0,"unit_price":2.0,"unit":"l","title":"Oatly Semi Oat Drink","size_g":1000.0},{"shop":"Sains","price":2.1,"unit_price":2.1,"unit":"l","title":"Oatly Oat Drink Semi Chilled 1L","size_g":1000.0},{"shop":"Tesco","price":1.5,"unit_price":1.5,"unit":"l","title":"Oatly Semi Oat Chilled Drink 1L","size_g":1000.0}]},{"id":23,"name":"Kenco Rich Instant Coffee Refill 150g","category":"drinks","confidence":0.95,"issues":["category_conflict"],"items":[{"shop":"ASDA","price":4.75,"unit_price":31.7,"unit":"kg","title":"Kenco Rich Instant Coffee Refill","size_g":150.0},{"shop":"Morrisons","price":4.75,"unit_price":31.7,"unit":"kg","title":"Kenco Rich Refill Instant Coffee","size_g":150.0},{"shop":"Sains","price":5.25,"unit_price":35.0,"unit":"kg","title":"Kenco Rich Instant Coffee Refill 150g","size_g":150.0},{"shop":"Tesco","price":4.65,"unit_price":31.0,"unit":"kg","title":"Kenco Rich Instant Coffee Refill 150G","size_g":150.0}]},{"id":4387,"name":"Halo Top Sea Salt Caramel Ice Cream 473ml","category":"food_cupboard","confidence":0.95,"issues":["category_conflict"],"items":[{"shop":"ASDA","price":3.75,"unit_price":7.9,"unit":"l","title":"Halo Top Sea Salt Caramel Ice Cream","size_g":473.0},{"shop":"Morrisons","price":5.25,"unit_price":11.1,"unit":"l","title":"Halo Top Sea Salt Caramel Ice Cream","size_g":473.0},{"shop":"Sains","price":5.25,"unit_price":11.1,"unit":"l","title":"Halo Top Sea Salt Caramel Ice Cream 473ml","size_g":473.0},{"shop":"Tesco","price":3.75,"unit_price":7.9,"unit":"l","title":"Halo Top Sea Salt Caramel Ice Cream 473Ml","size_g":473.0}]},{"id":4299,"name":"Sardines In Tomato Sauce (120g)","category":"food_cupboard","confidence":0.939,"issues":[],"items":[{"shop":"ASDA","price":0.55,"unit_price":4.6,"unit":"kg","title":"ASDA Sardines in Tomato Sauce","size_g":120.0},{"shop":"Morrisons","price":0.55,"unit_price":4.6,"unit":"kg","title":"Morrisons Sardines In Tomato Sauce (120g)","size_g":120.0},{"shop":"Sains","price":0.6,"unit_price":5.0,"unit":"kg","title":"Sainsbury's Sardines in Tomato Sauce 120g","size_g":120.0},{"shop":"Tesco","price":0.47,"unit_price":3.9,"unit":"kg","title":"Tesco Sardines In Tomato Sauce 120G","size_g":120.0}]},{"id":5803,"name":"Hovis Granary Thick Sliced Wholemeal Bread 800g","category":"free_from","confidence":0.936,"issues":["category_conflict"],"items":[{"shop":"ASDA","price":1.85,"unit_price":2.3,"unit":"kg","title":"Hovis Granary Wholemeal Bread","size_g":800.0},{"shop":"Morrisons","price":1.85,"unit_price":2.3,"unit":"kg","title":"Hovis Granary Wholemeal Bread","size_g":800.0},{"shop":"Sains","price":1.85,"unit_price":2.3,"unit":"kg","title":"Hovis Granary Thick Sliced Wholemeal Bread 800g","size_g":800.0},{"shop":"Tesco","price":1.85,"unit_price":2.3,"unit":"kg","title":"Hovis Granary Wholemeal Bread 800G","size_g":800.0}]},{"id":7536,"name":"19 Crimes The Uprising Red Wine","category":"drinks","confidence":0.932,"issues":[],"items":[{"shop":"ASDA","price":9.5,"unit_price":126.64,"unit":"l","title":"19 Crimes Red Wine","size_g":750.0},{"shop":"Morrisons","price":9.0,"unit_price":11.97,"unit":"l","title":"19 Crimes The Uprising Red Wine","size_g":750.0},{"shop":"Sains","price":8.0,"unit_price":10.64,"unit":"l","title":"19 Crimes Red Wine 75cl","size_g":750.0},{"shop":"Tesco","price":9.5,"unit_price":12.64,"unit":"l","title":"19 Crimes Red Wine 75Cl","size_g":750.0}]},{"id":9807,"name":"Squeaky Bean Applewood Smoked Ham Style Slices 80g","category":"fresh_food","confidence":0.93,"issues":["category_conflict"],"items":[{"shop":"ASDA","price":2.0,"unit_price":25.0,"unit":"kg","title":"Squeaky Bean Applewood Smoked Ham Style Slices","size_g":80.0},{"shop":"Morrisons","price":2.49,"unit_price":31.1,"unit":"kg","title":"Squeaky Bean Smoked Ham Slices","size_g":80.0},{"shop":"Sains","price":2.0,"unit_price":25.0,"unit":"kg","title":"Squeaky Bean Applewood Smoked Ham Style Slices 80g","size_g":80.0},{"shop":"Tesco","price":2.5,"unit_price":31.25,"unit":"kg","title":"Squeaky Bean Applewood Smoked Ham Style Slices 80G","size_g":80.0}]},{"id":4491,"name":"Pizza Express Garlic & Herb Pizza Dipping Sauce 255G","category":"food_cupboard","confidence":0.92,"issues":["size_minor"],"items":[{"shop":"ASDA","price":2.25,"unit_price":8.8,"unit":"kg","title":"Pizza Express Pizza Dipping Sauce Garlic & Herb","size_g":250.0},{"shop":"Morrisons","price":2.25,"unit_price":7.8,"unit":"kg","title":"Pizza Express Garlic And Herb Pizza Dipping Sauce","size_g":288.5},{"shop":"Sains","price":2.0,"unit_price":7.8,"unit":"kg","title":"Pizza Express Garlic & Herb Dipping Sauce 255g","size_g":255.0},{"shop":"Tesco","price":1.5,"unit_price":5.9,"unit":"kg","title":"Pizza Express Garlic & Herb Pizza Dipping Sauce 255G","size_g":255.0}]},{"id":2292,"name":"Vermouth Bianco 100c","category":"drinks","confidence":0.903,"issues":[],"items":[{"shop":"ASDA","price":6.8,"unit_price":6.8,"unit":"l","title":"Vermouth Bianco 100c","size_g":1000.0},{"shop":"Morrisons","price":8.0,"unit_price":7.98,"unit":"l","title":"Morrisons Vermouth Bianco","size_g":1000.0},{"shop":"Sains","price":8.0,"unit_price":7.98,"unit":"l","title":"Sainsbury's Vermouth Bianco 1L","size_g":1000.0},{"shop":"Tesco","price":8.0,"unit_price":7.98,"unit":"l","title":"Tesco Vermouth Bianco 1Ltr","size_g":1000.0}]},{"id":6294,"name":"Mr. Freeze Jubbly Cola Ice Lollies 8x62ml","category":"drinks","confidence":0.897,"issues":["category_conflict"],"items":[{"shop":"ASDA","price":1.6,"unit_price":3.2,"unit":"l","title":"Mr Freeze Jubbly Cola Ice Lollies","size_g":500.0},{"shop":"Morrisons","price":1.69,"unit_price":3.4,"unit":"l","title":"Jubbly Cola Ice Lollies","size_g":500.0},{"shop":"Sains","price":1.65,"unit_price":3.3,"unit":"l","title":"Mr. Freeze Jubbly Cola Ice Lollies 8x62ml","size_g":496.0},{"shop":"Tesco","price":1.65,"unit_price":3.3,"unit":"l","title":"Mr Freeze Jubbly Ice Lollies Cola 8X62ml","size_g":496.0}]},{"id":17084,"name":"Doritos Tortilla Chips Chilli Heatwave Sharing Bag Crisps 180g","category":"food_cupboard","confidence":0.879,"issues":[],"items":[{"shop":"ASDA","price":2.5,"unit_price":13.9,"unit":"kg","title":"Doritos Chilli Heatwave Sharing Tortilla Chips Crisps 180g","size_g":180.0},{"shop":"Morrisons","price":1.5,"unit_price":8.3,"unit":"kg","title":"Doritos Chilli Heatwave Sharing\u2026","size_g":180.0},{"shop":"Sains","price":1.5,"unit_price":8.3,"unit":"kg","title":"Doritos Chilli Heatwave Sharing Tortilla Chips Crisps 180g","size_g":180.0},{"shop":"Tesco","price":2.5,"unit_price":13.9,"unit":"kg","title":"Doritos Tortilla Chips Chilli Heatwave Sharing Bag Crisps 180g","size_g":180.0}]},{"id":8276,"name":"Brewdog Neon Dream Tropical 4 X 330Ml","category":"drinks","confidence":0.879,"issues":[],"items":[{"shop":"ASDA","price":6.0,"unit_price":4.55,"unit":"l","title":"BrewDog Neon Dream","size_g":1318.7},{"shop":"Morrisons","price":6.25,"unit_price":4.73,"unit":"l","title":"BrewDog Neon Dream Beer Cans","size_g":1321.4},{"shop":"Sains","price":5.25,"unit_price":3.97,"unit":"l","title":"BrewDog Neon Dream 4x330ml","size_g":1320.0},{"shop":"Tesco","price":5.25,"unit_price":3.98,"unit":"l","title":"Brewdog Neon Dream Tropical 4 X 330Ml","size_g":1320.0}]},{"id":14390,"name":"Soreen Apple Lunchbox Loaves Snack Bars","category":"free_from","confidence":0.867,"issues":["category_conflict"],"items":[{"shop":"ASDA","price":1.75,"unit_price":0.35,"unit":"unit","title":"Soreen Apple Lunchbox Loaves Snack Bars","size_g":null},{"shop":"Morrisons","price":1.25,"unit_price":0.25,"unit":"unit","title":"Soreen Lunchbox Loaves","size_g":null},{"shop":"Sains","price":1.8,"unit_price":0.36,"unit":"unit","title":"Soreen Apple Lunchbox Loaves 5x30g","size_g":150.0},{"shop":"Tesco","price":1.25,"unit_price":8.4,"unit":"kg","title":"Soreen 5 Apple Lunchbox Loaves 150G","size_g":150.0}]},{"id":2643,"name":"Aspall Suffolk Premier Cru Cyder Bottle","category":"drinks","confidence":0.866,"issues":[],"items":[{"shop":"ASDA","price":2.75,"unit_price":5.5,"unit":"l","title":"Aspall Premier Cru Cyder","size_g":500.0},{"shop":"Morrisons","price":2.75,"unit_price":5.5,"unit":"l","title":"Aspall Suffolk Premier Cru Cyder Bottle","size_g":500.0},{"shop":"Sains","price":2.85,"unit_price":5.7,"unit":"l","title":"Aspall Premier Cru Dry Cyder 500ml","size_g":500.0},{"shop":"Tesco","price":2.75,"unit_price":5.5,"unit":"l","title":"Aspall Premier Cru 500Ml","size_g":500.0}]},{"id":6510,"name":"Mexican Style Smoky BBQ Fajita Kit","category":"food_cupboard","confidence":0.857,"issues":[],"items":[{"shop":"ASDA","price":1.68,"unit_price":1.68,"unit":"unit","title":"ASDA Mexican Style Smoky BBQ Fajita Kit","size_g":null},{"shop":"Morrisons","price":2.49,"unit_price":5.2,"unit":"kg","title":"Morrisons Smoky BBQ Fajita Kit","size_g":475.0},{"shop":"Sains","price":1.79,"unit_price":3.6,"unit":"kg","title":"Sainsbury's Fajita Kit, Smoky BBQ 500g","size_g":500.0},{"shop":"Tesco","price":1.89,"unit_price":3.98,"unit":"kg","title":"Tesco Smoky Bbq Fajita Kit 475G","size_g":475.0}]},{"id":1938,"name":"Pot Noodle Original Curry Instant Noodles 90g","category":"food_cupboard","confidence":0.846,"issues":[],"items":[{"shop":"ASDA","price":0.65,"unit_price":7.22,"unit":"kg","title":"Pot Noodle Original Curry","size_g":90.0},{"shop":"Morrisons","price":1.29,"unit_price":14.33,"unit":"kg","title":"Pot Noodle Original Curry Standard","size_g":90.0},{"shop":"Sains","price":1.3,"unit_price":14.44,"unit":"kg","title":"Pot Noodle Original Curry 90g","size_g":90.0},{"shop":"Tesco","price":1.2,"unit_price":13.33,"unit":"kg","title":"Pot Noodle Original Curry Instant Noodles 90g","size_g":90.0}]},{"id":8733,"name":"Magnum Classic Chocolate Ice Cream Sticks 6x100ml","category":"frozen","confidence":0.843,"issues":[],"items":[{"shop":"ASDA","price":4.5,"unit_price":7.5,"unit":"l","title":"Magnum Classic Ice Cream","size_g":600.0},{"shop":"Morrisons","price":5.0,"unit_price":9.1,"unit":"l","title":"Magnum Mini Ice Cream Sticks","size_g":549.5},{"shop":"Sains","price":4.5,"unit_price":7.5,"unit":"l","title":"Magnum Classic Chocolate Ice Cream Sticks 6x100ml","size_g":600.0},{"shop":"Tesco","price":4.5,"unit_price":7.5,"unit":"l","title":"Magnum Classic Ice Cream Sticks 6x100ml","size_g":600.0}]},{"id":3330,"name":"Yeo Valley Organic Greek Style Natural Yogurt 950g","category":"fresh_food","confidence":0.778,"issues":["category_conflict"],"items":[{"shop":"ASDA","price":3.4,"unit_price":3.6,"unit":"kg","title":"Yeo Valley Organic Greek Style Natural Yogurt","size_g":950.0},{"shop":"Morrisons","price":3.85,"unit_price":4.1,"unit":"kg","title":"Yeo Valley Organic Greek Style Yogurt","size_g":950.0},{"shop":"Sains","price":3.85,"unit_price":4.1,"unit":"kg","title":"Yeo Valley Organic Greek Style Natural Yogurt 950g","size_g":950.0},{"shop":"Tesco","price":4.5,"unit_price":4.7,"unit":"kg","title":"Yeo Valley Organic Natural Kefir Yogurt 950g","size_g":950.0}]},{"id":2177,"name":"The Fruit Factory Strawberry Fruit Hearts 5x20g","category":"fresh_food","confidence":0.746,"issues":["size_minor","category_conflict"],"items":[{"shop":"ASDA","price":2.3,"unit_price":23.0,"unit":"kg","title":"The Fruit Factory Strawberry Fruit Hearts","size_g":100.0},{"shop":"Morrisons","price":2.35,"unit_price":26.1,"unit":"kg","title":"The Fruit Factory Strawberry, App\u2026","size_g":90.0},{"shop":"Sains","price":2.5,"unit_price":25.0,"unit":"kg","title":"The Fruit Factory Strawberry Fruit Hearts 5x20g","size_g":100.0},{"shop":"Tesco","price":2.45,"unit_price":24.5,"unit":"kg","title":"Fruit Factory Fruit Hearts Strawberry 5X20g","size_g":100.0}]},{"id":9597,"name":"Delicious Cheese & Onion Quiche","category":"fresh_food","confidence":0.743,"issues":[],"items":[{"shop":"ASDA","price":2.0,"unit_price":5.0,"unit":"kg","title":"ASDA Delicious Cheese & Onion Quiche","size_g":400.0},{"shop":"Morrisons","price":2.75,"unit_price":6.9,"unit":"kg","title":"Morrisons Cheese & Onion Quiche","size_g":400.0},{"shop":"Sains","price":3.0,"unit_price":7.5,"unit":"kg","title":"Sainsbury's Cheese & Onion Quiche 400g","size_g":400.0},{"shop":"Tesco","price":2.25,"unit_price":5.6,"unit":"kg","title":"Tesco Cheddar & Onion Quiche 400G","size_g":400.0}]},{"id":15641,"name":"Timothy Taylor's Landlord Strong Pale Ale Bottle","category":"drinks","confidence":0.738,"issues":["brand_conflict"],"items":[{"shop":"ASDA","price":2.3,"unit_price":4.6,"unit":"l","title":"Timothy Taylor's Landlord Classic Pale Ale","size_g":500.0},{"shop":"Morrisons","price":2.25,"unit_price":4.5,"unit":"l","title":"Timothy Taylor's Landlord Strong Pale Ale Bottle","size_g":500.0},{"shop":"Sains","price":2.25,"unit_price":4.5,"unit":"l","title":"Timothy Taylor's Landlord Ale 500ml","size_g":500.0},{"shop":"Tesco","price":2.25,"unit_price":4.5,"unit":"l","title":"Timothy Taylors Landlord Pale 500Ml","size_g":500.0}]},{"id":8921,"name":"MSC Cod Fishcakes x2 270g","category":"fresh_food","confidence":0.728,"issues":["category_conflict"],"items":[{"shop":"ASDA","price":1.29,"unit_price":4.78,"unit":"kg","title":"ASDA Creamy 2 Cod Fishcakes","size_g":270.0},{"shop":"Morrisons","price":1.79,"unit_price":6.63,"unit":"kg","title":"Morrisons Cod Fillet Fishcakes","size_g":270.0},{"shop":"Sains","price":1.8,"unit_price":6.67,"unit":"kg","title":"Sainsbury's MSC Cod Fishcakes x2 270g","size_g":270.0},{"shop":"Tesco","price":1.8,"unit_price":6.67,"unit":"kg","title":"Tesco 2 Cod Fishcakes 270G","size_g":270.0}]},{"id":2822,"name":"Linda McCartney's Vegetarian Lincolnshire Sausages x6 300g","category":"frozen","confidence":0.727,"issues":["size_minor","category_conflict"],"items":[{"shop":"ASDA","price":2.3,"unit_price":8.52,"unit":"kg","title":"Linda McCartney's 6 Vegetarian Sausages","size_g":270.0},{"shop":"Morrisons","price":2.5,"unit_price":9.26,"unit":"kg","title":"Linda McCartney's 6 Vegetarian Sausages","size_g":270.0},{"shop":"Sains","price":2.0,"unit_price":6.66,"unit":"kg","title":"Linda McCartney's Vegetarian Lincolnshire Sausages x6 300g","size_g":300.0},{"shop":"Tesco","price":2.5,"unit_price":9.26,"unit":"kg","title":"Linda Mccartney 6 Vegetarian Sausages 270G","size_g":270.0}]},{"id":11239,"name":"Mr Kipling Cherry Bakewells 30% Less Sugar 6 Pack","category":"other","confidence":0.708,"issues":[],"items":[{"shop":"ASDA","price":2.1,"unit_price":11.05,"unit":"kg","title":"ASDA Free From 4 Cherry Bakewells Cakes","size_g":190.0},{"shop":"Morrisons","price":2.0,"unit_price":0.33,"unit":"unit","title":"Mr Kipling Cherry Bakewells","size_g":null},{"shop":"Sains","price":1.5,"unit_price":0.25,"unit":"unit","title":"Mr Kipling Cherry Bakewells Cakes x6","size_g":null},{"shop":"Tesco","price":2.4,"unit_price":0.4,"unit":"unit","title":"Mr Kipling Cherry Bakewells 30% Less Sugar 6 Pack","size_g":null}]},{"id":5419,"name":"Pringles Sizzl'N Kickin' Sour Cream Sharing Crisps 180g","category":"food_cupboard","confidence":0.703,"issues":[],"items":[{"shop":"ASDA","price":1.85,"unit_price":10.0,"unit":"kg","title":"Pringles Paprika Sharing Crisps","size_g":180.0},{"shop":"Morrisons","price":1.85,"unit_price":10.0,"unit":"kg","title":"Pringles Paprika Sharing Crisps","size_g":180.0},{"shop":"Sains","price":2.25,"unit_price":12.2,"unit":"kg","title":"Pringles Paprika Sharing Crisps 185g","size_g":185.0},{"shop":"Tesco","price":2.25,"unit_price":12.5,"unit":"kg","title":"Pringles Sizzl'N Kickin' Sour Cream Sharing Crisps 180g","size_g":180.0}]},{"id":16146,"name":"Hot Paprika 50g","category":"food_cupboard","confidence":0.651,"issues":["size_minor"],"items":[{"shop":"ASDA","price":1.0,"unit_price":20.0,"unit":"kg","title":"ASDA Hot Paprika 50g","size_g":50.0},{"shop":"Morrisons","price":1.09,"unit_price":24.0,"unit":"kg","title":"Morrisons Paprika","size_g":45.4},{"shop":"Sains","price":1.1,"unit_price":25.0,"unit":"kg","title":"Sainsbury's Hot Paprika 44g","size_g":44.0},{"shop":"Tesco","price":1.0,"unit_price":19.0,"unit":"kg","title":"Tesco Paprika 52G","size_g":52.0}]},{"id":11070,"name":"Thatchers Haze Cloudy Somerset Cider 10x440ml","category":"drinks","confidence":0.644,"issues":[],"items":[{"shop":"ASDA","price":9.5,"unit_price":2.16,"unit":"l","title":"Thatchers Haze Cloudy Somerset Cider 10 Pack","size_g":4398.1},{"shop":"Morrisons","price":10.5,"unit_price":2.39,"unit":"l","title":"Thatchers Haze Cloudy Somerset Cider Cans","size_g":4393.3},{"shop":"Sains","price":11.0,"unit_price":2.5,"unit":"l","title":"Thatchers Haze Cloudy Somerset Cider 10x440ml","size_g":4400.0},{"shop":"Tesco","price":9.5,"unit_price":2.16,"unit":"l","title":"Thatchers Gold Cider 10X440ml Can","size_g":4400.0}]},{"id":2601,"name":"Nice Drop Trebbiano Pinot Grigio 75cl","category":"drinks","confidence":0.644,"issues":[],"items":[{"shop":"ASDA","price":4.25,"unit_price":56.65,"unit":"l","title":"Nice Drop Trebbiano Pinot Grigio 75cl","size_g":750.0},{"shop":"Morrisons","price":7.5,"unit_price":9.98,"unit":"l","title":"Primo Vere Pinot Grigio","size_g":750.0},{"shop":"Sains","price":5.0,"unit_price":6.65,"unit":"l","title":"San Marco Trebbiano Pinot Grigio 75cl","size_g":750.0},{"shop":"Tesco","price":6.0,"unit_price":7.98,"unit":"l","title":"Dino Trebbiano Pinot Grigio 75Cl","size_g":750.0}]},{"id":12042,"name":"Delicately Sweet Tenderheart Cabbage","category":"fresh_food","confidence":0.621,"issues":["low_word_overlap"],"items":[{"shop":"ASDA","price":0.67,"unit_price":0.67,"unit":"unit","title":"ASDA Delicately Sweet Tenderheart Cabbage","size_g":null},{"shop":"Morrisons","price":0.7,"unit_price":0.7,"unit":"unit","title":"Morrisons Sweetheart Cabbage","size_g":null},{"shop":"Sains","price":0.67,"unit_price":0.67,"unit":"unit","title":"Sainsbury's Sweetheart Cabbage","size_g":null},{"shop":"Tesco","price":0.67,"unit_price":0.67,"unit":"unit","title":"Tesco Sweetheart Cabbage Each","size_g":null}]},{"id":2381,"name":"Jura Single Malt Scotch Whisky Pale Ale, Cask Edition 70cl","category":"drinks","confidence":0.617,"issues":[],"items":[{"shop":"ASDA","price":27.0,"unit_price":38.57,"unit":"l","title":"Jura Single Malt Scotch Whisky Pale Ale Cask Edition","size_g":700.0},{"shop":"Morrisons","price":40.5,"unit_price":57.86,"unit":"l","title":"Jura Aged 10 Years Single Malt\u2026","size_g":700.0},{"shop":"Sains","price":39.0,"unit_price":55.71,"unit":"l","title":"Jura Single Malt Scotch Whisky Pale Ale, Cask Edition 70cl","size_g":700.0},{"shop":"Tesco","price":37.0,"unit_price":52.86,"unit":"l","title":"Jura Bourbon Cask Single Malt Scotch Whisky 70Cl","size_g":700.0}]},{"id":1438,"name":"Wholefoods Bulgar Wheat 500G","category":"food_cupboard","confidence":0.595,"issues":["low_word_overlap"],"items":[{"shop":"ASDA","price":2.0,"unit_price":4.0,"unit":"kg","title":"ASDA Bulgur Wheat","size_g":500.0},{"shop":"Morrisons","price":1.89,"unit_price":3.78,"unit":"kg","title":"Morrisons Wholefoods Bulgur Wheat","size_g":500.0},{"shop":"Sains","price":1.9,"unit_price":3.8,"unit":"kg","title":"Sainsbury's Cracked Bulgur Wheat 500g","size_g":500.0},{"shop":"Tesco","price":2.0,"unit_price":4.0,"unit":"kg","title":"Tesco Wholefoods Bulgar Wheat 500G","size_g":500.0}]},{"id":3737,"name":"Deliciously Free From Cornflakes 300g","category":"food_cupboard","confidence":0.57,"issues":["category_conflict","low_word_overlap"],"items":[{"shop":"ASDA","price":1.75,"unit_price":5.8,"unit":"kg","title":"ASDA Free From Corn Flakes","size_g":300.0},{"shop":"Morrisons","price":1.95,"unit_price":6.5,"unit":"kg","title":"Morrisons Free From Corn Flakes","size_g":300.0},{"shop":"Sains","price":1.75,"unit_price":5.8,"unit":"kg","title":"Sainsbury's Deliciously Free From Cornflakes 300g","size_g":300.0},{"shop":"Tesco","price":2.1,"unit_price":7.0,"unit":"kg","title":"Tesco Free From Branflakes 300G","size_g":300.0}]},{"id":7568,"name":"Freixenet Italian Sparkling Ros\u00e9 Small Wine Bottle 20cl","category":"drinks","confidence":0.504,"issues":["low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":7.0,"unit_price":7.0,"unit":"unit","title":"Freixenet Italian Sparkling Ros\u00e9 and Reed Diffuser Set","size_g":null},{"shop":"Morrisons","price":3.75,"unit_price":18.7,"unit":"l","title":"Freixenet Italian Rose Sparkling Wine","size_g":200.0},{"shop":"Sains","price":3.75,"unit_price":18.7,"unit":"l","title":"Freixenet Italian Sparkling Ros\u00e9 Small Wine Bottle 20cl","size_g":200.0},{"shop":"Tesco","price":2.5,"unit_price":12.48,"unit":"l","title":"Canvino Naturally Sparkling Rose Wine 200Ml","size_g":200.0}]},{"id":11721,"name":"British Pork & Apple Sausages, Taste the Difference x6 400g","category":"other","confidence":0.498,"issues":["category_conflict","low_text_similarity"],"items":[{"shop":"ASDA","price":3.25,"unit_price":8.12,"unit":"kg","title":"ASDA Extra Special 6 Pork & Fennel Sausages 400g","size_g":400.0},{"shop":"Morrisons","price":3.25,"unit_price":8.13,"unit":"kg","title":"Morrisons The Best Pork & Ale Sausages","size_g":400.0},{"shop":"Sains","price":3.25,"unit_price":8.13,"unit":"kg","title":"Sainsbury's British Pork & Apple Sausages, Taste the Difference x6 400g","size_g":400.0},{"shop":"Tesco","price":0.58,"unit_price":1.43,"unit":"kg","title":"Stockwell & Co Baked Beans & Pork Sausages 405G","size_g":405.0}]},{"id":2592,"name":"Grande Nuit Sauvignon Blanc 750Ml","category":"drinks","confidence":0.45,"issues":["low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":6.75,"unit_price":89.98,"unit":"l","title":"Graham Norton Sauvignon Blanc","size_g":750.0},{"shop":"Morrisons","price":8.75,"unit_price":11.64,"unit":"l","title":"Beefsteak Club Mendoza Malbec","size_g":750.0},{"shop":"Sains","price":6.25,"unit_price":8.31,"unit":"l","title":"Star Gazer Sauvignon Blanc 75cl","size_g":750.0},{"shop":"Tesco","price":7.5,"unit_price":9.98,"unit":"l","title":"Grande Nuit Sauvignon Blanc 750Ml","size_g":750.0}]},{"id":10691,"name":"Carotino Red Palm Fruit & Rapeseed Oil 500ml","category":"food_cupboard","confidence":0.45,"issues":["low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":2.0,"unit_price":4.0,"unit":"l","title":"Carotino Healthier Oil for Cooking","size_g":500.0},{"shop":"Morrisons","price":2.5,"unit_price":5.0,"unit":"l","title":"Carotino Red Palm Fruit & Rapeseed Oil","size_g":500.0},{"shop":"Sains","price":2.5,"unit_price":5.0,"unit":"l","title":"Carotino Red Palm Fruit & Rapeseed Oil 500ml","size_g":500.0},{"shop":"Tesco","price":2.0,"unit_price":4.0,"unit":"kg","title":"Aasani Red Skin Peanut Kernels 500G","size_g":500.0}]},{"id":8429,"name":"Wirra Wirra Church Block Cabernet Sauvignon Shiraz Merlot 750ml","category":"drinks","confidence":0.45,"issues":["low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":11.5,"unit_price":153.29,"unit":"l","title":"Wirra Wirra Church Block Cabernet Sauvignon-Shiraz-Merlot","size_g":750.0},{"shop":"Morrisons","price":8.5,"unit_price":11.3,"unit":"l","title":"Carta Roja Pura Jumilla Organic Wine","size_g":750.0},{"shop":"Sains","price":13.0,"unit_price":17.29,"unit":"l","title":"Wirra Wirra Church Block Cabernet Sauvignon Shiraz Merlot 750ml","size_g":750.0},{"shop":"Tesco","price":4.39,"unit_price":5.84,"unit":"l","title":"Lateral Chilean Cabernet Sauvignon 75Cl","size_g":750.0}]},{"id":2666,"name":"Berne Inspiration C\u00f4tes de Provence 75cl","category":"drinks","confidence":0.45,"issues":["low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":11.5,"unit_price":153.29,"unit":"l","title":"Berne Intemporelle C\u00f4tes de Provence","size_g":750.0},{"shop":"Morrisons","price":8.0,"unit_price":10.64,"unit":"l","title":"Morrisons The Best Vinho Verde 'Loureiro' Wine","size_g":750.0},{"shop":"Sains","price":15.0,"unit_price":19.95,"unit":"l","title":"Berne Inspiration C\u00f4tes de Provence 75cl","size_g":750.0},{"shop":"Tesco","price":11.0,"unit_price":14.63,"unit":"l","title":"Dv Catena Cabernet Franc Historico 750Ml","size_g":750.0}]},{"id":1070,"name":"Crazy Jack Organic Soft Ready to Eat Figs 200g","category":"food_cupboard","confidence":0.432,"issues":["size_minor","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":2.0,"unit_price":11.43,"unit":"kg","title":"Whitworths Soft Figs","size_g":170.0},{"shop":"Morrisons","price":2.0,"unit_price":11.4,"unit":"kg","title":"Whitworths Figs","size_g":175.0},{"shop":"Sains","price":1.6,"unit_price":9.1,"unit":"kg","title":"Whitworths Figs 175g","size_g":175.0},{"shop":"Tesco","price":2.5,"unit_price":12.5,"unit":"kg","title":"Crazy Jack Organic Soft Ready to Eat Figs 200g","size_g":200.0}]},{"id":5187,"name":"Fresh Extra Thick Double Cream 300ml","category":"fresh_food","confidence":0.423,"issues":["category_conflict","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":1.5,"unit_price":5.0,"unit":"l","title":"ASDA Fresh Extra Thick Double Cream 300ml","size_g":300.0},{"shop":"Morrisons","price":1.35,"unit_price":4.5,"unit":"l","title":"Morrisons British Whipping Cream","size_g":300.0},{"shop":"Sains","price":0.99,"unit_price":3.3,"unit":"l","title":"Sainsbury's British Single Cream 300ml","size_g":300.0},{"shop":"Tesco","price":1.35,"unit_price":4.5,"unit":"l","title":"Tesco British Whipping Cream 300Ml","size_g":300.0}]},{"id":4537,"name":"Heinz Pickle Flavour Tomato Ketchup 400ml","category":"food_cupboard","confidence":0.414,"issues":["size_minor","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":3.0,"unit_price":7.5,"unit":"l","title":"Heinz Pickle Flavour Tomato Ketchup 400ml","size_g":400.0},{"shop":"Morrisons","price":3.4,"unit_price":8.5,"unit":"l","title":"Heinz Pickle Flavour Tomato Ketchup","size_g":400.0},{"shop":"Sains","price":1.8,"unit_price":4.62,"unit":"kg","title":"Heinz Beanz Cheesy 390g","size_g":390.0},{"shop":"Tesco","price":2.5,"unit_price":7.3,"unit":"kg","title":"Heinz Tomato Ketchup Bottle 342G","size_g":342.0}]},{"id":15449,"name":"Magicorn Eazypop Microwave Popcorn Sweet Flavour 85G","category":"food_cupboard","confidence":0.405,"issues":["size_disagree","low_text_similarity"],"items":[{"shop":"ASDA","price":1.1,"unit_price":11.0,"unit":"kg","title":"ASDA Sweet & Salty Popcorn","size_g":100.0},{"shop":"Morrisons","price":1.1,"unit_price":11.0,"unit":"kg","title":"Morrisons Sweet & Salt Popcorn","size_g":100.0},{"shop":"Sains","price":1.5,"unit_price":13.6,"unit":"kg","title":"Sainsbury's Simply Sweet Butterfly Popcorn 110g","size_g":110.0},{"shop":"Tesco","price":0.9,"unit_price":10.6,"unit":"kg","title":"Magicorn Eazypop Microwave Popcorn Sweet Flavour 85G","size_g":85.0}]},{"id":12312,"name":"The BAKERY at ASDA Rice Pudding","category":"bakery","confidence":0.4,"issues":["category_conflict","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":3.15,"unit_price":7.9,"unit":"kg","title":"The BAKERY at ASDA Rice Pudding","size_g":400.0},{"shop":"Morrisons","price":1.89,"unit_price":4.7,"unit":"kg","title":"Morrisons Buttercream Frosting","size_g":400.0},{"shop":"Sains","price":0.9,"unit_price":2.3,"unit":"kg","title":"Sainsbury's Rice Pudding, Creamed 400g","size_g":400.0},{"shop":"Tesco","price":0.9,"unit_price":2.2,"unit":"kg","title":"Tesco Rice Pudding 400G","size_g":400.0}]},{"id":48,"name":"Taylors of Harrogate Yorkshire Tea Black Loose Leaf Tea","category":"drinks","confidence":0.4,"issues":["category_conflict","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":2.85,"unit_price":11.4,"unit":"kg","title":"Taylors of Harrogate Yorkshire Tea Black Loose Leaf Tea","size_g":250.0},{"shop":"Morrisons","price":3.29,"unit_price":13.2,"unit":"kg","title":"Taylors Of Harrogate Yorkshire Tea\u2026","size_g":250.0},{"shop":"Sains","price":1.65,"unit_price":6.6,"unit":"kg","title":"Sainsbury's Red Label Loose Tea 250g","size_g":250.0},{"shop":"Tesco","price":3.3,"unit_price":13.2,"unit":"kg","title":"Yorkshire 80 Teabags 250G","size_g":250.0}]},{"id":3715,"name":"Nestl\u00e9 Nesquik Chocolate Cereal 375g","category":"food_cupboard","confidence":0.37,"issues":["size_minor","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":3.29,"unit_price":7.8,"unit":"kg","title":"Weetos Chocolatey Hoops","size_g":425.0},{"shop":"Morrisons","price":3.49,"unit_price":8.3,"unit":"kg","title":"Weetos Chocolatey Hoops Cereal","size_g":425.0},{"shop":"Sains","price":3.3,"unit_price":7.9,"unit":"kg","title":"Weetos Chocolatey Hoops 420g","size_g":420.0},{"shop":"Tesco","price":2.49,"unit_price":6.6,"unit":"kg","title":"Nestl\u00e9 Nesquik Chocolate Cereal 375g","size_g":375.0}]},{"id":3615,"name":"Properoni Pepperoni Hot Paprika Sliced 80g","category":"fresh_food","confidence":0.366,"issues":["size_minor","category_conflict","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":1.3,"unit_price":17.3,"unit":"kg","title":"Csabah\u00fas Pepperoni Snack Hot","size_g":75.0},{"shop":"Morrisons","price":1.49,"unit_price":19.9,"unit":"kg","title":"Csabahus Pepperoni Snack Hot","size_g":75.0},{"shop":"Sains","price":1.1,"unit_price":12.94,"unit":"l","title":"Grace Hot Pepper Sauce 85ml","size_g":85.0},{"shop":"Tesco","price":3.0,"unit_price":37.5,"unit":"kg","title":"Properoni Pepperoni Hot Paprika Sliced 80g","size_g":80.0}]},{"id":1635,"name":"Ambrosia Low Fat Custard Pots 4x125g","category":"food_cupboard","confidence":0.332,"issues":["size_disagree","category_conflict","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":1.3,"unit_price":10.4,"unit":"kg","title":"ASDA Free From Custard Creams","size_g":125.0},{"shop":"Morrisons","price":1.5,"unit_price":12.0,"unit":"kg","title":"Morrisons Free From Custard Creams","size_g":125.0},{"shop":"Sains","price":2.5,"unit_price":5.0,"unit":"kg","title":"Ambrosia Low Fat Custard Pots 4x125g","size_g":500.0},{"shop":"Tesco","price":1.3,"unit_price":10.4,"unit":"kg","title":"Tesco Free From Custard Creams 125G","size_g":125.0}]},{"id":8158,"name":"San Miguel Premium Lager Beer Cans 4x568ml","category":"drinks","confidence":0.323,"issues":["size_disagree","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":2.5,"unit_price":3.85,"unit":"l","title":"Kingfisher Premium Lager Beer","size_g":650.0},{"shop":"Morrisons","price":2.5,"unit_price":3.85,"unit":"l","title":"Kingfisher Lager","size_g":650.0},{"shop":"Sains","price":7.0,"unit_price":3.08,"unit":"l","title":"San Miguel Premium Lager Beer Cans 4x568ml","size_g":2272.0},{"shop":"Tesco","price":2.5,"unit_price":3.85,"unit":"l","title":"Kingfisher Premium Lager 650Ml","size_g":650.0}]},{"id":9276,"name":"Gosh! Aromatic Moroccan Falafel 266G","category":"fresh_food","confidence":0.32,"issues":["size_minor","category_conflict","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":5.0,"unit_price":17.61,"unit":"kg","title":"ASDA Sticky 4 Teriyaki Basa Kebabs 284g","size_g":284.0},{"shop":"Morrisons","price":3.09,"unit_price":11.6,"unit":"kg","title":"Gosh! Moroccan Falafel","size_g":270.0},{"shop":"Sains","price":1.7,"unit_price":5.67,"unit":"kg","title":"Sainsbury's Hot & Spicy Stir Fry 300g","size_g":300.0},{"shop":"Tesco","price":3.1,"unit_price":11.65,"unit":"kg","title":"Gosh! Aromatic Moroccan Falafel 266G","size_g":266.0}]},{"id":683,"name":"Moroccan Cous Cous, Taste the Difference 200g","category":"fresh_food","confidence":0.32,"issues":["size_minor","category_conflict","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":1.95,"unit_price":10.8,"unit":"kg","title":"Ginsters Vegan Moroccan Vegetable Pasty","size_g":180.0},{"shop":"Morrisons","price":1.69,"unit_price":8.4,"unit":"kg","title":"Morrisons Free From Chocola\u2026","size_g":200.0},{"shop":"Sains","price":1.65,"unit_price":8.3,"unit":"kg","title":"Sainsbury's Moroccan Cous Cous, Taste the Difference 200g","size_g":200.0},{"shop":"Tesco","price":1.25,"unit_price":6.9,"unit":"kg","title":"Ginsters Vegan Moroccan Vegetable Pasty 180G","size_g":180.0}]},{"id":12226,"name":"Plant Based by ASDA 6 Meat-Free Chorizo Inspired Sausages 270g","category":"free_from","confidence":0.32,"issues":["size_minor","category_conflict","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":2.25,"unit_price":8.33,"unit":"kg","title":"Plant Based by ASDA 6 Meat-Free Chorizo Inspired Sausages 270g","size_g":270.0},{"shop":"Morrisons","price":3.45,"unit_price":11.5,"unit":"kg","title":"Tarczynski Kielbasa Geesowska Polish Sausage","size_g":300.0},{"shop":"Sains","price":3.45,"unit_price":11.5,"unit":"kg","title":"Tarczynski Geesowska Sausage 300g","size_g":300.0},{"shop":"Tesco","price":3.7,"unit_price":12.3,"unit":"kg","title":"Tarczynski Geesowska Polish Sausages 300G","size_g":300.0}]},{"id":1386,"name":"Jamie Oliver Microwave Ready To Eat Ultimate Black Daal 250g","category":"food_cupboard","confidence":0.265,"issues":["size_minor","brand_conflict","low_text_similarity","low_word_overlap"],"items":[{"shop":"ASDA","price":1.4,"unit_price":6.36,"unit":"kg","title":"Ben's Original Mixed Pepper Microwave Rice","size_g":220.0},{"shop":"Morrisons","price":1.39,"unit_price":6.3,"unit":"kg","title":"Bens Original Mixed Pepper Microwave Rice","size_g":220.0},{"shop":"Sains","price":1.75,"unit_price":7.0,"unit":"kg","title":"Jamie Oliver Microwave Ready To Eat Ultimate Black Daal 250g","size_g":250.0},{"shop":"Tesco","price":3.4,"unit_price":13.6,"unit":"kg","title":"Tesco Mixed Nuts 250G","size_g":250.0}]}];

// === Theme ===
const SHOPS = {
  ASDA:      { full: "ASDA",          color: "#3F8624", bg: "#EAF3DD" },
  Tesco:     { full: "Tesco",         color: "#0E548E", bg: "#E0EAF3" },
  Sains:     { full: "Sainsbury's",   color: "#E5751F", bg: "#FBEBD9" },
  Morrisons: { full: "Morrisons",     color: "#005B2D", bg: "#DBE8DD" },
};
const SHOP_ORDER = ["ASDA", "Tesco", "Sains", "Morrisons"];

// === Confidence model ===
// We grade each cluster on a continuous 0-1 score and bucket it into 5 tiers,
// each with a verbal label that reflects what we'd actually tell a user.
function bucketOf(conf) {
  if (conf >= 0.85) return "very_high";
  if (conf >= 0.70) return "high";
  if (conf >= 0.55) return "moderate";
  if (conf >= 0.40) return "low";
  return "very_low";
}

const BUCKET_META = {
  very_high: {
    label: "Confirmed",
    desc:  "Same product across all retailers",
    color: "#1F6F3F", bg: "#E1F0E5", icon: ShieldCheck,
  },
  high: {
    label: "Likely match",
    desc:  "Strong evidence these are the same",
    color: "#3F6E4D", bg: "#E5EEDF", icon: ShieldCheck,
  },
  moderate: {
    label: "Probably similar",
    desc:  "Could be the same; some differences in wording",
    color: "#8B5E1A", bg: "#F4E7CF", icon: Shield,
  },
  low: {
    label: "Possible substitute",
    desc:  "Related products, but not identical",
    color: "#A0421F", bg: "#F1DAD0", icon: ShieldAlert,
  },
  very_low: {
    label: "Uncertain",
    desc:  "We're not sure these are comparable",
    color: "#9E3B2A", bg: "#F2DCD4", icon: AlertTriangle,
  },
};

// Map issue tags into human-readable explanations
const ISSUE_TEXT = {
  size_disagree:        "pack sizes differ by more than 20%",
  size_minor:           "pack sizes differ slightly",
  brand_conflict:       "products list different brand names",
  category_conflict:    "products are listed in different categories",
  low_text_similarity:  "product names share little wording",
  low_word_overlap:     "few keywords overlap between names",
};

const formatGBP = (n) => `£${n.toFixed(2)}`;

// =============================================================================
// Confidence chip + meter
// =============================================================================
function ConfidenceChip({ conf }) {
  const m = BUCKET_META[bucketOf(conf)];
  const Icon = m.icon;
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider"
      style={{ background: m.bg, color: m.color, letterSpacing: "0.06em" }}
    >
      <Icon size={12} strokeWidth={2.5} />
      {m.label}
    </span>
  );
}

function ConfidenceMeter({ conf, showLabel = true }) {
  const m = BUCKET_META[bucketOf(conf)];
  const pct = Math.round(conf * 100);
  return (
    <div className="flex items-center gap-2">
      {showLabel && (
        <span className="text-[10px] uppercase tracking-wider font-semibold" style={{ color: m.color }}>
          {pct}%
        </span>
      )}
      <div className="flex-1 h-1.5 rounded-full overflow-hidden bg-stone-200 min-w-[60px]">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: m.color }} />
      </div>
    </div>
  );
}

// =============================================================================
// Header
// =============================================================================
function Header({ basketCount, onCheckout, view }) {
  return (
    <header className="border-b border-stone-300 bg-[#FAF7F2] sticky top-0 z-40 backdrop-blur supports-[backdrop-filter]:bg-[#FAF7F2]/90">
      <div className="max-w-6xl mx-auto px-5 py-4 flex items-center justify-between">
        <div className="flex items-baseline gap-3">
          <h1 className="font-serif text-2xl tracking-tight" style={{ fontFamily: "'Fraunces', Georgia, serif", fontWeight: 600 }}>
            Shop<span style={{ color: "#C44536" }}>Wiser</span>
          </h1>
          <span className="hidden sm:block text-[11px] text-stone-500 italic">
            grocery comparison with honest uncertainty
          </span>
        </div>
        <button
          onClick={onCheckout}
          className="flex items-center gap-2 text-sm font-medium px-3 py-2 rounded-full bg-stone-900 text-stone-50 hover:bg-stone-700 transition-colors"
        >
          <span>{view === "compare" ? "Browse" : "Compare"}</span>
          <span className="bg-[#C44536] text-white text-[11px] rounded-full w-5 h-5 inline-flex items-center justify-center font-semibold">
            {basketCount}
          </span>
        </button>
      </div>
    </header>
  );
}

// =============================================================================
// Browse view: search + filter by minimum confidence
// =============================================================================
function BrowseView({ basket, addToBasket, removeFromBasket, minConf, setMinConf, onOpenItem }) {
  const [query, setQuery] = useState("");
  const [activeCat, setActiveCat] = useState("all");

  const categories = useMemo(() => {
    const cats = new Set(DEMO_DATA.map((d) => d.category));
    return ["all", ...Array.from(cats).filter((c) => c !== "other").sort()];
  }, []);

  const filtered = useMemo(() => {
    let pool = DEMO_DATA.filter((d) => d.confidence >= minConf);
    if (activeCat !== "all") pool = pool.filter((d) => d.category === activeCat);
    if (query.trim()) {
      const q = query.toLowerCase();
      pool = pool.filter((d) => d.name.toLowerCase().includes(q));
    }
    return pool;
  }, [query, activeCat, minConf]);

  const inBasketIds = new Set(basket.map((b) => b.id));
  const totalCount = DEMO_DATA.length;
  const visibleCount = filtered.length;

  return (
    <section className="px-5 py-6 max-w-6xl mx-auto">
      <div className="mb-5">
        <h2
          className="font-serif text-3xl mb-1.5 leading-tight"
          style={{ fontFamily: "'Fraunces', Georgia, serif", fontWeight: 500 }}
        >
          Build your basket.
        </h2>
        <p className="text-stone-600 text-sm leading-relaxed max-w-2xl">
          Every product is matched across the four major UK supermarkets with a
          confidence score we can defend. If the match is uncertain, we tell you
          before you commit, not after you've checked out somewhere cheaper than
          expected.
        </p>
      </div>

      {/* Search */}
      <div className="relative mb-4">
        <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-stone-400" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search 'cheddar', 'pasta', 'crisps'..."
          className="w-full pl-11 pr-4 py-3.5 bg-white border border-stone-300 rounded-2xl text-sm focus:outline-none focus:border-stone-900 focus:ring-2 focus:ring-stone-900/10 transition-all"
        />
      </div>

      {/* Confidence filter — the user-facing precision/coverage toggle */}
      <ConfidenceFilter minConf={minConf} setMinConf={setMinConf} totalCount={totalCount} visibleCount={visibleCount} />

      {/* Category chips */}
      <div className="flex gap-2 overflow-x-auto pb-3 mb-5 -mx-5 px-5 scrollbar-hide">
        {categories.map((c) => (
          <button
            key={c}
            onClick={() => setActiveCat(c)}
            className={`whitespace-nowrap px-3.5 py-1.5 rounded-full text-xs font-medium transition-all ${
              activeCat === c ? "bg-stone-900 text-stone-50" : "bg-stone-100 text-stone-600 hover:bg-stone-200"
            }`}
          >
            {c.replace("_", " ")}
          </button>
        ))}
      </div>

      {/* Cards */}
      <div className="grid sm:grid-cols-2 gap-3">
        {filtered.map((item) => {
          const inBasket = inBasketIds.has(item.id);
          const minPrice = Math.min(...item.items.map((i) => i.price));
          return (
            <div
              key={item.id}
              className="bg-white border border-stone-200 rounded-2xl p-4 hover:border-stone-400 hover:shadow-sm transition-all cursor-pointer"
              onClick={() => onOpenItem(item)}
            >
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex-1 min-w-0">
                  <h3
                    className="font-serif text-base leading-snug mb-2"
                    style={{ fontFamily: "'Fraunces', Georgia, serif", fontWeight: 500 }}
                  >
                    {item.name}
                  </h3>
                  <div className="flex items-center gap-2 mb-2">
                    <ConfidenceChip conf={item.confidence} />
                    <span className="text-[11px] text-stone-500">from {formatGBP(minPrice)}</span>
                  </div>
                  <ConfidenceMeter conf={item.confidence} />
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (inBasket) removeFromBasket(item.id);
                    else addToBasket(item);
                  }}
                  className={`shrink-0 w-9 h-9 rounded-full flex items-center justify-center transition-all ${
                    inBasket ? "bg-[#C44536] text-white" : "bg-stone-100 text-stone-700 hover:bg-stone-900 hover:text-white"
                  }`}
                >
                  {inBasket ? <Minus size={16} /> : <Plus size={16} />}
                </button>
              </div>
              <div className="flex gap-1.5 mt-3">
                {SHOP_ORDER.map((s) => {
                  const has = item.items.find((i) => i.shop === s);
                  const isMin = has && has.price === minPrice;
                  return (
                    <div
                      key={s}
                      className="flex-1 text-center py-1.5 rounded-lg text-[10px] font-semibold tracking-wide"
                      style={{
                        background: has ? SHOPS[s].bg : "#F5F2ED",
                        color: has ? SHOPS[s].color : "#C2BAB0",
                        outline: isMin ? `1.5px solid ${SHOPS[s].color}` : "none",
                      }}
                    >
                      {has ? formatGBP(has.price) : "—"}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-16 text-stone-500">
          <Filter size={32} className="mx-auto mb-3 text-stone-300" strokeWidth={1.5} />
          <p className="text-sm">
            No matches above {Math.round(minConf * 100)}% confidence.{" "}
            {minConf > 0 && (
              <button onClick={() => setMinConf(0)} className="underline">
                Lower the threshold to see more.
              </button>
            )}
          </p>
        </div>
      )}
    </section>
  );
}

// =============================================================================
// Confidence filter — embeds the precision-coverage tradeoff in the UI
// =============================================================================
function ConfidenceFilter({ minConf, setMinConf, totalCount, visibleCount }) {
  const presets = [
    { v: 0.0,  label: "All matches",        sub: "include uncertain" },
    { v: 0.55, label: "Probable or better", sub: "balanced" },
    { v: 0.70, label: "Likely or better",   sub: "stricter" },
    { v: 0.85, label: "Confirmed only",     sub: "highest trust" },
  ];
  return (
    <div className="bg-white border border-stone-200 rounded-2xl p-4 mb-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Filter size={14} className="text-stone-500" />
          <span className="text-xs font-semibold tracking-wide uppercase text-stone-700">
            Match-quality filter
          </span>
        </div>
        <span className="text-[11px] text-stone-500">
          showing <strong className="text-stone-800">{visibleCount}</strong> of {totalCount}
        </span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {presets.map((p) => {
          const active = Math.abs(minConf - p.v) < 0.01;
          return (
            <button
              key={p.v}
              onClick={() => setMinConf(p.v)}
              className={`text-left px-3 py-2 rounded-xl border-2 transition-all ${
                active
                  ? "border-stone-900 bg-stone-900 text-white"
                  : "border-stone-200 hover:border-stone-400 bg-white"
              }`}
            >
              <div className="text-[12px] font-semibold leading-tight">{p.label}</div>
              <div className={`text-[10px] mt-0.5 ${active ? "text-stone-300" : "text-stone-500"}`}>
                {p.sub}
              </div>
            </button>
          );
        })}
      </div>
      <p className="text-[11px] text-stone-500 italic mt-3 leading-relaxed">
        Stricter filter, fewer products but higher confidence each is genuinely the same. Looser filter, more
        products but more chance one is a near-substitute.
      </p>
    </div>
  );
}

// =============================================================================
// Compare view: per-retailer totals + per-item breakdown + uncertainty banner
// =============================================================================
function CompareView({ basket, removeFromBasket, openDetail }) {
  const totals = useMemo(() => {
    const t = {};
    SHOP_ORDER.forEach((s) => (t[s] = { total: 0, missing: 0, lowConfTotal: 0, items: [] }));
    basket.forEach((it) => {
      SHOP_ORDER.forEach((s) => {
        const stock = it.items.find((x) => x.shop === s);
        if (stock) {
          t[s].total += stock.price;
          t[s].items.push({ ...stock, _it: it });
          if (it.confidence < 0.55) t[s].lowConfTotal += stock.price;
        } else {
          t[s].missing += 1;
        }
      });
    });
    return t;
  }, [basket]);

  const cheapest = useMemo(() => {
    let best = null;
    SHOP_ORDER.forEach((s) => {
      if (totals[s].missing === 0 && (best === null || totals[s].total < totals[best].total)) {
        best = s;
      }
    });
    return best;
  }, [totals]);

  const avgConfidence = useMemo(() => {
    if (basket.length === 0) return 1;
    return basket.reduce((s, b) => s + b.confidence, 0) / basket.length;
  }, [basket]);

  const lowConfItems = basket.filter((b) => b.confidence < 0.55);
  const veryLowConfItems = basket.filter((b) => b.confidence < 0.40);

  if (basket.length === 0) {
    return (
      <section className="px-5 py-16 max-w-6xl mx-auto text-center">
        <Shield className="mx-auto mb-4 text-stone-300" size={40} strokeWidth={1.5} />
        <p
          className="font-serif text-2xl mb-2 text-stone-700"
          style={{ fontFamily: "'Fraunces', Georgia, serif" }}
        >
          Your basket is empty.
        </p>
        <p className="text-sm text-stone-500">Browse to add items.</p>
      </section>
    );
  }

  return (
    <section className="px-5 py-6 max-w-6xl mx-auto">
      <h2
        className="font-serif text-3xl mb-2"
        style={{ fontFamily: "'Fraunces', Georgia, serif", fontWeight: 500 }}
      >
        The verdict.
      </h2>
      <p className="text-sm text-stone-600 mb-5 leading-relaxed">
        Below is what your {basket.length}-item basket costs at each retailer, alongside how much we trust the
        comparison.
      </p>

      {/* Trust banner */}
      <TrustBanner avg={avgConfidence} basket={basket} lowCount={lowConfItems.length} veryLowCount={veryLowConfItems.length} />

      {/* Per-shop totals */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
        {SHOP_ORDER.map((s) => {
          const t = totals[s];
          const isWinner = s === cheapest;
          const unavailable = t.missing > 0;
          const trustedTotal = t.total - t.lowConfTotal;
          return (
            <div
              key={s}
              className={`rounded-2xl p-4 border-2 transition-all ${
                isWinner ? "border-[#C44536] bg-[#FFF8F4]" : "border-stone-200 bg-white"
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-[11px] font-bold tracking-widest uppercase" style={{ color: SHOPS[s].color }}>
                  {SHOPS[s].full}
                </span>
                {isWinner && (
                  <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-[#C44536] text-white">
                    cheapest
                  </span>
                )}
              </div>
              <div
                className="font-serif text-3xl mb-1"
                style={{ fontFamily: "'Fraunces', Georgia, serif", fontWeight: 600 }}
              >
                {formatGBP(t.total)}
              </div>
              <div className="text-[11px] text-stone-500">
                {unavailable ? (
                  <span className="text-amber-700">{t.missing} not stocked</span>
                ) : t.lowConfTotal > 0 ? (
                  <>
                    <span className="text-stone-700 font-medium">{formatGBP(trustedTotal)}</span>{" "}
                    trusted, {formatGBP(t.lowConfTotal)} uncertain
                  </>
                ) : (
                  <span className="text-stone-400">all items confirmed</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Item breakdown */}
      <h3
        className="font-serif text-xl mb-3 text-stone-800"
        style={{ fontFamily: "'Fraunces', Georgia, serif" }}
      >
        Item by item
      </h3>
      <div className="bg-white rounded-2xl border border-stone-200 overflow-hidden">
        {basket.map((item, idx) => {
          const valid = item.items;
          const minP = valid.length ? Math.min(...valid.map((v) => v.price)) : null;
          return (
            <div
              key={item.id}
              className={`p-4 flex items-center gap-4 hover:bg-stone-50 cursor-pointer transition-colors ${
                idx > 0 ? "border-t border-stone-200" : ""
              }`}
              onClick={() => openDetail(item)}
            >
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  removeFromBasket(item.id);
                }}
                className="w-7 h-7 rounded-full bg-stone-100 hover:bg-stone-200 text-stone-500 inline-flex items-center justify-center shrink-0"
              >
                <X size={14} />
              </button>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                  <h4 className="font-medium text-sm leading-tight">{item.name}</h4>
                </div>
                <div className="flex items-center gap-3">
                  <ConfidenceChip conf={item.confidence} />
                  <div className="flex-1 max-w-[140px]">
                    <ConfidenceMeter conf={item.confidence} showLabel={false} />
                  </div>
                </div>
              </div>
              <div className="hidden sm:flex gap-1.5">
                {SHOP_ORDER.map((s) => {
                  const p = item.items.find((i) => i.shop === s);
                  if (!p)
                    return (
                      <div key={s} className="w-16 text-center py-1.5 rounded-lg text-[11px] text-stone-300 bg-stone-50">
                        —
                      </div>
                    );
                  const isMin = p.price === minP;
                  return (
                    <div
                      key={s}
                      className={`w-16 text-center py-1.5 rounded-lg text-[11px] font-semibold ${
                        isMin ? "bg-[#FFE8E0] text-[#9E3B2A]" : "bg-stone-50 text-stone-700"
                      }`}
                    >
                      {formatGBP(p.price)}
                    </div>
                  );
                })}
              </div>
              <ArrowRight size={16} className="text-stone-400 shrink-0" />
            </div>
          );
        })}
      </div>
    </section>
  );
}

// =============================================================================
// Trust banner: visualises overall basket confidence
// =============================================================================
function TrustBanner({ avg, basket, lowCount, veryLowCount }) {
  const m = BUCKET_META[bucketOf(avg)];
  const Icon = m.icon;
  return (
    <div
      className="rounded-2xl p-4 mb-6 border-l-4"
      style={{ background: m.bg, borderColor: m.color }}
    >
      <div className="flex items-start gap-3">
        <Icon size={20} className="shrink-0 mt-0.5" style={{ color: m.color }} strokeWidth={2} />
        <div className="flex-1">
          <div className="flex items-baseline gap-3 mb-1">
            <span
              className="font-semibold text-sm"
              style={{ color: m.color }}
            >
              Overall basket trust: {Math.round(avg * 100)}%
            </span>
          </div>
          <p className="text-xs text-stone-700 leading-relaxed">
            {veryLowCount > 0 && (
              <>
                <strong style={{ color: m.color }}>{veryLowCount} item{veryLowCount > 1 ? "s" : ""}</strong>{" "}
                in your basket {veryLowCount > 1 ? "are" : "is"} matched with very low confidence — we cannot
                guarantee {veryLowCount > 1 ? "they're" : "it's"} comparable across retailers. The cheapest-supermarket
                verdict could be wrong.{" "}
              </>
            )}
            {veryLowCount === 0 && lowCount > 0 && (
              <>
                {lowCount} item{lowCount > 1 ? "s have" : " has"} low match confidence. Tap to inspect before
                trusting the price difference.
              </>
            )}
            {veryLowCount === 0 && lowCount === 0 && (
              <>All items are matched with high confidence. The price differences shown reflect like-for-like comparisons.</>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Detail modal: explain the model's reasoning
// =============================================================================
function DetailPanel({ item, onClose }) {
  const [expanded, setExpanded] = useState(false);
  if (!item) return null;
  const m = BUCKET_META[bucketOf(item.confidence)];
  const Icon = m.icon;

  // Detect concrete differences in titles
  const titles = item.items.map((i) => i.title.toLowerCase());
  const sizes = item.items.map((i) => i.size_g).filter(Boolean);
  const sizeRange = sizes.length >= 2 ? Math.max(...sizes) / Math.min(...sizes) : 1;

  return (
    <div
      className="fixed inset-0 bg-black/40 z-50 flex items-end sm:items-center justify-center p-0 sm:p-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-[#FAF7F2] w-full sm:max-w-2xl rounded-t-3xl sm:rounded-3xl max-h-[92vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1 min-w-0 pr-4">
              <h3
                className="font-serif text-xl leading-tight mb-2"
                style={{ fontFamily: "'Fraunces', Georgia, serif", fontWeight: 500 }}
              >
                {item.name}
              </h3>
              <ConfidenceChip conf={item.confidence} />
            </div>
            <button
              onClick={onClose}
              className="w-9 h-9 rounded-full bg-stone-100 hover:bg-stone-200 inline-flex items-center justify-center"
            >
              <X size={18} />
            </button>
          </div>

          {/* Confidence panel with score, meter, and explanation */}
          <div
            className="rounded-xl px-4 py-3.5 mb-5 border"
            style={{ background: m.bg, borderColor: m.color + "40" }}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Icon size={16} style={{ color: m.color }} />
                <span className="text-xs font-bold uppercase tracking-wider" style={{ color: m.color }}>
                  {m.label}
                </span>
              </div>
              <span className="font-serif text-lg" style={{ fontFamily: "'Fraunces', Georgia, serif", color: m.color, fontWeight: 600 }}>
                {Math.round(item.confidence * 100)}%
              </span>
            </div>
            <div className="mb-2">
              <ConfidenceMeter conf={item.confidence} showLabel={false} />
            </div>
            <p className="text-xs leading-relaxed" style={{ color: m.color }}>
              {m.desc}
              {item.issues && item.issues.length > 0 && ":"}
            </p>
            {item.issues && item.issues.length > 0 && (
              <ul className="mt-1.5 text-[11px] space-y-0.5" style={{ color: m.color }}>
                {item.issues.map((i) => (
                  <li key={i}>• {ISSUE_TEXT[i] || i}</li>
                ))}
              </ul>
            )}
            {!item.issues || item.issues.length === 0 ? (
              <p className="text-[11px] mt-1.5" style={{ color: m.color }}>
                Names, brand, pack size and category all agree across retailers.
              </p>
            ) : null}
          </div>

          {/* Per-retailer cards */}
          <div className="space-y-2.5 mb-4">
            {SHOP_ORDER.map((s) => {
              const it = item.items.find((i) => i.shop === s);
              if (!it) {
                return (
                  <div key={s} className="rounded-xl px-4 py-3 bg-stone-100 text-stone-400 flex items-center justify-between">
                    <span className="text-xs font-bold tracking-wider uppercase">{SHOPS[s].full}</span>
                    <span className="text-xs italic">Not stocked</span>
                  </div>
                );
              }
              return (
                <div key={s} className="rounded-xl px-4 py-3 border border-stone-200 bg-white">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[10px] font-bold tracking-widest uppercase" style={{ color: SHOPS[s].color }}>
                      {SHOPS[s].full}
                    </span>
                    <span
                      className="font-serif text-xl"
                      style={{ fontFamily: "'Fraunces', Georgia, serif", fontWeight: 600 }}
                    >
                      {formatGBP(it.price)}
                    </span>
                  </div>
                  <p className="text-xs text-stone-700 leading-snug">{it.title}</p>
                  {it.unit_price && (
                    <p className="text-[11px] text-stone-500 mt-1">
                      {formatGBP(it.unit_price)}/{it.unit}
                      {it.size_g && ` · ${it.size_g}g`}
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          {/* Optional: model transparency */}
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-[11px] text-stone-500 flex items-center gap-1.5 hover:text-stone-800"
          >
            {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            How the confidence score is computed
          </button>
          {expanded && (
            <div className="mt-3 text-[11px] text-stone-600 leading-relaxed bg-white p-3 rounded-xl border border-stone-200">
              <p className="mb-2">
                We combine five signals across the matched products: text similarity (30%), word overlap (25%), pack size
                agreement (20%), brand agreement (15%) and category agreement (10%). The score above is the weighted
                blend.
              </p>
              <p>
                In production this would be the calibrated output of a learned ranker, so that "70% confidence" really means
                the match is correct 70% of the time on a held-out test set. Our current implementation is the heuristic
                approximation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// App
// =============================================================================
export default function App() {
  const [view, setView] = useState("browse");
  const [basket, setBasket] = useState(() => {
    // Seed with mixed-confidence items so the uncertainty story is visible from the start
    const high = DEMO_DATA.filter((d) => d.confidence >= 0.85);
    const mid = DEMO_DATA.filter((d) => d.confidence >= 0.55 && d.confidence < 0.85);
    const low = DEMO_DATA.filter((d) => d.confidence < 0.55);
    const seed = [];
    if (high[0]) seed.push(high[0]);
    if (high[2]) seed.push(high[2]);
    if (mid[0]) seed.push(mid[0]);
    if (low[0]) seed.push(low[0]);
    return seed;
  });
  const [minConf, setMinConf] = useState(0.0);
  const [detail, setDetail] = useState(null);

  const addToBasket = (item) => setBasket((b) => (b.find((x) => x.id === item.id) ? b : [...b, item]));
  const removeFromBasket = (id) => setBasket((b) => b.filter((x) => x.id !== id));

  // Load fonts
  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href =
      "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=DM+Sans:wght@400;500;600;700&display=swap";
    document.head.appendChild(link);
    return () => document.head.removeChild(link);
  }, []);

  return (
    <div
      className="min-h-screen"
      style={{ background: "#FAF7F2", fontFamily: "'DM Sans', system-ui, sans-serif", color: "#1A1A1A" }}
    >
      <Header
        basketCount={basket.length}
        view={view}
        onCheckout={() => setView(view === "compare" ? "browse" : "compare")}
      />

      <nav className="border-b border-stone-200 bg-[#FAF7F2]">
        <div className="max-w-6xl mx-auto px-5 flex gap-6">
          <button
            onClick={() => setView("browse")}
            className={`py-3 text-sm font-medium tracking-wide transition-all border-b-2 ${
              view === "browse"
                ? "border-[#C44536] text-stone-900"
                : "border-transparent text-stone-500 hover:text-stone-800"
            }`}
          >
            Browse & build
          </button>
          <button
            onClick={() => setView("compare")}
            className={`py-3 text-sm font-medium tracking-wide transition-all border-b-2 ${
              view === "compare"
                ? "border-[#C44536] text-stone-900"
                : "border-transparent text-stone-500 hover:text-stone-800"
            }`}
          >
            Compare ({basket.length})
          </button>
        </div>
      </nav>

      {view === "browse" ? (
        <BrowseView
          basket={basket}
          addToBasket={addToBasket}
          removeFromBasket={removeFromBasket}
          minConf={minConf}
          setMinConf={setMinConf}
          onOpenItem={setDetail}
        />
      ) : (
        <CompareView basket={basket} removeFromBasket={removeFromBasket} openDetail={setDetail} />
      )}

      <DetailPanel item={detail} onClose={() => setDetail(null)} />

      <footer className="max-w-6xl mx-auto px-5 py-8 mt-12 border-t border-stone-200 text-[11px] text-stone-500 leading-relaxed">
        <div className="flex items-start gap-2">
          <Info size={12} className="mt-0.5 shrink-0" />
          <p>
            Demonstration of how confidence-aware matching could surface in the user-facing product. Every product in
            this demo carries a real confidence score derived from text similarity, word overlap, pack-size agreement
            and brand/category agreement. In a production pipeline this score would be the calibrated output of a
            learned ranker.
          </p>
        </div>
      </footer>
    </div>
  );
}
