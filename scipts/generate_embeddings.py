"""
Embedding Generation Script for Hybrid Product Matching
========================================================

This script generates semantic embeddings and feature vectors for all products.
Run this ONCE to create the embedding cache, then use matching-hybrid.ipynb.

Architecture:
- Semantic embeddings: SentenceTransformer (all-MiniLM-L6-v2)
- Feature vectors: TF-IDF for token-level matching
- Metadata: Units, categories, brands for hard constraints

Output:
- data/embeddings/semantic_embeddings.npy: Dense semantic vectors
- data/embeddings/tfidf_features.npz: Sparse TF-IDF vectors
- data/embeddings/product_metadata.pkl: Product info for matching
- data/embeddings/embedding_index.pkl: Mapping indices to product IDs
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
BATCH_SIZE = 256  # For embedding generation
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TFIDF_MAX_FEATURES = 5000
OUTPUT_DIR = Path("data/embeddings")

print("="*100)
print("HYBRID MATCHING: EMBEDDING GENERATION")
print("="*100)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n✓ Output directory: {OUTPUT_DIR}")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print(f"\n{'='*100}")
print("STEP 1: LOADING DATA")
print("="*100)

df = pd.read_csv("data/normalized_products.csv")
print(f"\n✓ Loaded {len(df):,} products")

# Basic data validation
required_cols = ['supermarket', 'normalized_name', 'core_product_name', 
                 'unit_value', 'unit_type', 'category', 'known_brand', 
                 'tier_type', 'pack_quantity']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print(f"✓ All required columns present")

# Fill missing values
df['unit_value'] = df['unit_value'].fillna(0)
df['unit_type'] = df['unit_type'].fillna('none')
df['known_brand'] = df['known_brand'].fillna('none')
df['tier_type'] = df['tier_type'].fillna('none')
df['normalized_name'] = df['normalized_name'].fillna('')
df['core_product_name'] = df['core_product_name'].fillna('')
df['pack_quantity'] = df['pack_quantity'].fillna(1)

# Filter out poor quality data
df = df[df['normalized_name'].str.len() >= 3].copy()
df = df.reset_index(drop=True)

print(f"✓ After filtering: {len(df):,} products")

# Supermarket distribution
print(f"\nSupermarket distribution:")
for sm in ['ASDA', 'Morrisons', 'Sains', 'Tesco']:
    count = (df['supermarket'] == sm).sum()
    print(f"  {sm:10s}: {count:6,} products ({count/len(df)*100:5.1f}%)")

# ============================================================================
# 2. GENERATE SEMANTIC EMBEDDINGS
# ============================================================================
print(f"\n{'='*100}")
print("STEP 2: GENERATING SEMANTIC EMBEDDINGS")
print("="*100)

print(f"\nLoading SentenceTransformer model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✓ Model loaded | Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Create enhanced text for embedding (combines multiple fields for better semantic understanding)
def create_embedding_text(row):
    """
    Create rich text representation for semantic embedding.
    Combines normalized name, brand, and category for better matching.
    """
    parts = [row['normalized_name']]
    
    # Add brand if known
    if row['known_brand'] != 'none':
        parts.append(row['known_brand'])
    
    # Add category for context
    if pd.notna(row['category']):
        parts.append(row['category'])
    
    return ' '.join(parts)

print("\nCreating embedding texts...")
df['embedding_text'] = df.apply(create_embedding_text, axis=1)
embedding_texts = df['embedding_text'].tolist()

print(f"Sample embedding texts:")
for i in range(min(3, len(embedding_texts))):
    print(f"  {i+1}. {embedding_texts[i][:70]}")

# Generate embeddings in batches
print(f"\nGenerating embeddings for {len(embedding_texts):,} products...")
print(f"Batch size: {BATCH_SIZE}")

embeddings = []
for i in tqdm(range(0, len(embedding_texts), BATCH_SIZE), desc="Encoding"):
    batch = embedding_texts[i:i+BATCH_SIZE]
    batch_embeddings = model.encode(batch, 
                                     show_progress_bar=False,
                                     convert_to_numpy=True,
                                     normalize_embeddings=True)  # L2 normalize for cosine similarity
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings)
print(f"\n✓ Generated embeddings: shape {embeddings.shape}")
print(f"  Embedding dimension: {embeddings.shape[1]}")
print(f"  Memory size: {embeddings.nbytes / 1024 / 1024:.1f} MB")

# Save embeddings
embeddings_path = OUTPUT_DIR / "semantic_embeddings.npy"
np.save(embeddings_path, embeddings)
print(f"✓ Saved to: {embeddings_path}")

# ============================================================================
# 3. GENERATE TF-IDF FEATURES
# ============================================================================
print(f"\n{'='*100}")
print("STEP 3: GENERATING TF-IDF FEATURES")
print("="*100)

print(f"\nCreating TF-IDF vectorizer (max features: {TFIDF_MAX_FEATURES})")

# Use both word-level and character n-grams for robust matching
vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=(1, 2),          # Unigrams and bigrams
    analyzer='word',
    lowercase=True,
    min_df=2,                     # Ignore very rare terms
    max_df=0.95,                  # Ignore very common terms
    sublinear_tf=True             # Use log scaling for term frequency
)

print("Fitting TF-IDF vectorizer...")
# Use normalized_name for TF-IDF (token-based matching)
tfidf_features = vectorizer.fit_transform(df['normalized_name'])

print(f"\n✓ TF-IDF features generated: shape {tfidf_features.shape}")
print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
print(f"  Sparsity: {(1 - tfidf_features.nnz / (tfidf_features.shape[0] * tfidf_features.shape[1]))*100:.2f}%")
print(f"  Memory size: {tfidf_features.data.nbytes / 1024 / 1024:.1f} MB")

# Save TF-IDF features and vectorizer
from scipy.sparse import save_npz
tfidf_path = OUTPUT_DIR / "tfidf_features.npz"
save_npz(tfidf_path, tfidf_features)
print(f"✓ Saved TF-IDF features to: {tfidf_path}")

vectorizer_path = OUTPUT_DIR / "tfidf_vectorizer.pkl"
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"✓ Saved vectorizer to: {vectorizer_path}")

# ============================================================================
# 4. PREPARE METADATA FOR MATCHING
# ============================================================================
print(f"\n{'='*100}")
print("STEP 4: PREPARING METADATA")
print("="*100)

# Create metadata dictionary with all info needed for matching
metadata = {
    'supermarket': df['supermarket'].values,
    'category': df['category'].values,
    'unit_value': df['unit_value'].values,
    'unit_type': df['unit_type'].values,
    'known_brand': df['known_brand'].values,
    'tier_type': df['tier_type'].values,
    'pack_quantity': df['pack_quantity'].values,
    'normalized_name': df['normalized_name'].values,
    'core_product_name': df['core_product_name'].values,
    'original_index': df.index.values,
    'n_products': len(df)
}

# Add derived features
metadata['is_known_brand'] = (df['known_brand'] != 'none').values
metadata['has_unit'] = (df['unit_value'] > 0).values

print(f"\nMetadata prepared:")
print(f"  Products: {metadata['n_products']:,}")
print(f"  Known brands: {metadata['is_known_brand'].sum():,}")
print(f"  Products with units: {metadata['has_unit'].sum():,}")

# Save metadata
metadata_path = OUTPUT_DIR / "product_metadata.pkl"
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"\n✓ Saved metadata to: {metadata_path}")

# ============================================================================
# 5. CREATE EFFICIENT INDEX STRUCTURES
# ============================================================================
print(f"\n{'='*100}")
print("STEP 5: CREATING INDEX STRUCTURES")
print("="*100)

# Create index by category and supermarket for fast filtering
from collections import defaultdict

indices_by_category = defaultdict(list)
indices_by_supermarket = defaultdict(list)
indices_by_brand = defaultdict(list)

for idx, row in df.iterrows():
    indices_by_category[row['category']].append(idx)
    indices_by_supermarket[row['supermarket']].append(idx)
    if row['known_brand'] != 'none':
        indices_by_brand[row['known_brand']].append(idx)

index_structures = {
    'by_category': dict(indices_by_category),
    'by_supermarket': dict(indices_by_supermarket),
    'by_brand': dict(indices_by_brand),
}

print(f"\nIndex statistics:")
print(f"  Categories: {len(indices_by_category)}")
print(f"  Supermarkets: {len(indices_by_supermarket)}")
print(f"  Known brands: {len(indices_by_brand)}")

# Category sizes
print(f"\nProducts per category:")
for cat, indices in sorted(indices_by_category.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {cat:15s}: {len(indices):6,} products")

# Save indices
indices_path = OUTPUT_DIR / "index_structures.pkl"
with open(indices_path, 'wb') as f:
    pickle.dump(index_structures, f)
print(f"\n✓ Saved index structures to: {indices_path}")

# ============================================================================
# 6. VALIDATION & SUMMARY
# ============================================================================
print(f"\n{'='*100}")
print("STEP 6: VALIDATION & SUMMARY")
print("="*100)

# Verify all files exist and are readable
files_to_check = [
    ('Semantic embeddings', embeddings_path),
    ('TF-IDF features', tfidf_path),
    ('TF-IDF vectorizer', vectorizer_path),
    ('Product metadata', metadata_path),
    ('Index structures', indices_path)
]

print(f"\nValidating output files:")
all_valid = True
for name, path in files_to_check:
    if path.exists():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  ✓ {name:25s}: {size_mb:8.2f} MB")
    else:
        print(f"  ✗ {name:25s}: MISSING")
        all_valid = False

if not all_valid:
    raise RuntimeError("Some output files are missing!")

# Calculate theoretical maximum clusters
n_per_supermarket = df['supermarket'].value_counts()
theoretical_max = n_per_supermarket.min()
print(f"\nTheoretical maximum 4-way clusters:")
print(f"  Min products per supermarket: {theoretical_max:,}")
print(f"  Assuming perfect matching: ~{theoretical_max:,} 4-way clusters")
print(f"  Target (per specification): ~15,000 4-way clusters")

# Memory usage summary
total_size_mb = sum(path.stat().st_size for _, path in files_to_check) / 1024 / 1024
print(f"\nTotal disk space used: {total_size_mb:.1f} MB")

print(f"\n{'='*100}")
print("✅ EMBEDDING GENERATION COMPLETE")
print("="*100)
print(f"\nNext steps:")
print(f"  1. Open matching-hybrid.ipynb")
print(f"  2. Run the hybrid matching algorithm")
print(f"  3. Aim for ~15,000 high-quality 4-way clusters")
print(f"\nEmbedding cache location: {OUTPUT_DIR.absolute()}")
print("="*100)
