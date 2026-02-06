import pandas as pd
import re
import os
from quantulum3 import parser

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'data/raw.csv'
OUTPUT_FILE = 'data/processed_inventory.csv'

# Common Grocery Brands/Keywords to strip for "Clean Names"
NOISE_WORDS = [
    "tesco", "asda", "sainsbury's", "sainsburys", "morrisons", "waitrose", "aldi", "lidl",
    "value", "saver", "basics", "essential", "extra special", "finest", 
    "taste the difference", "best of", "by sainsbury's", "no added sugar",
    "organic", "free from", "gluten free"
]

class GroceryAttributeExtractor:
    """
    A hybrid extractor that combines Regex for structure (Multipacks)
    with Quantulum3 for NLP-based unit parsing.
    """
    
    def __init__(self):
        # Regex to catch "4 x" or "6x" patterns before we send the rest to Quantulum
        self.multipack_prefix = re.compile(r'(?P<count>\d+)\s*x\s*', re.IGNORECASE)

    def normalize_quantity(self, quantity):
        """
        Converts Quantulum object to a standard string (g or ml).
        Example: 1.5 kg -> 1500g
        """
        unit_name = quantity.unit.name
        value = quantity.value

        # Mass Handling
        if unit_name in ['gram', 'milligram', 'kilogram', 'pound', 'ounce']:
            if unit_name == 'kilogram':
                return f"{value * 1000:.0f}g"
            elif unit_name == 'milligram':
                return f"{value / 1000:.2f}g"
            elif unit_name == 'pound':
                return f"{value * 453.592:.0f}g"
            elif unit_name == 'ounce':
                return f"{value * 28.3495:.0f}g"
            else: # gram
                return f"{value:.0f}g"

        # Volume Handling
        elif unit_name in ['litre', 'millilitre', 'centilitre', 'pint', 'gallon']:
            if unit_name == 'litre':
                return f"{value * 1000:.0f}ml"
            elif unit_name == 'centilitre':
                return f"{value * 10:.0f}ml"
            elif unit_name == 'pint':
                return f"{value * 568.261:.0f}ml" # UK Pint
            else: # millilitre
                return f"{value:.0f}ml"
        
        # Count Handling (fallback)
        elif unit_name == 'dimensionless':
            return f"{int(value)}pk"

        return None

    def extract(self, text):
        if not isinstance(text, str):
            return "unknown"
        
        text = text.lower()
        multiplier = 1
        
        # 1. Check for Multipack prefix "4 x ..."
        # Quantulum handles "400g" well, but not "4x400g".
        # We strip the "4x", remember the 4, and let Quantulum parse the rest.
        multi_match = self.multipack_prefix.search(text)
        if multi_match:
            multiplier = int(multi_match.group('count'))
            # Remove the "4 x" part so Quantulum just sees "400g"
            text_to_parse = text.replace(multi_match.group(0), " ")
        else:
            text_to_parse = text

        try:
            # 2. Use Quantulum3 to parse the remaining text
            quants = parser.parse(text_to_parse)
            
            # Filter for Mass/Volume units only
            relevant_quants = [
                q for q in quants 
                if q.unit.entity.name in ['mass', 'volume'] 
                or q.unit.name == 'dimensionless' # catches "6 pack"
            ]

            if not relevant_quants:
                return "unknown"

            # 3. Selection Logic: 
            # If multiple weights found (e.g., "10g protein, 500g net"), 
            # we usually want the LAST one or the largest one.
            # Heuristic: The largest mass/volume is usually the product size.
            best_match = max(relevant_quants, key=lambda q: q.value)
            
            # Apply the multiplier (if we found a multipack)
            best_match.value = best_match.value * multiplier
            
            # Normalize to standard string
            result = self.normalize_quantity(best_match)
            return result if result else "unknown"

        except Exception as e:
            # Fallback for very messy data
            return "error"

def clean_title(text):
    """
    Normalizes product titles for clustering.
    Removes store names, 'value' brands, and special chars.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Remove noise words
    for word in NOISE_WORDS:
        text = text.replace(word, "")
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    return " ".join(text.split()).title()

def main():
    print(f"Loading data from {INPUT_FILE}...")
    
    # Check if file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Initialize Extractor
    extractor = GroceryAttributeExtractor()
    
    print("Extracting attributes (this may take a moment due to NLP parsing)...")
    # Apply extraction
    df['normalized_weight'] = df['names'].apply(extractor.extract)
    
    print("Cleaning product titles...")
    # Apply cleaning
    df['clean_name'] = df['names'].apply(clean_title)
    
    # Filter out failed extractions for a quick quality check
    success_rate = (df['normalized_weight'] != 'unknown').mean()
    print(f"Extraction Success Rate: {success_rate:.1%}")
    
    # Create a unique clustering key
    # (Clean Name + Weight is a proxy for a 'Product ID')
    df['cluster_key'] = df['clean_name'] + " " + df['normalized_weight']
    
    # Save Output
    print(f"Saving processed data to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Preview
    print("\nData Preview:")
    print(df[['names', 'clean_name', 'normalized_weight']].head(10))

if __name__ == "__main__":
    main()