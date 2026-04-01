"""Level A: Candidate Generation via Vector Search."""

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, TOP_K_CANDIDATES


def create_embedding_text(row) -> str:
    """Constructs the string to be vectorized (Level A1 in PDF)."""
    brand = str(row.get('known_brand_clean', ''))
    brand = brand if brand != 'nan' else ''

    name = str(row.get('normalized_name', ''))
    cat = str(row.get('category', ''))

    # Coarse size bucket for the text embedding
    size = ''
    uv = row.get('unit_value')
    ut = row.get('unit_type')
    if pd.notna(uv) and uv is not None:
        ut_s = '' if pd.isna(ut) else str(ut)
        size = f"{int(float(uv) // 50 * 50)}{ut_s}"

    return f"{brand} {name} {cat} {size}".strip()


def retrieve_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Builds FAISS indices per retailer and retrieves cross-retailer Top-K candidates."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    df = df.copy()
    df['embed_text'] = df.apply(create_embedding_text, axis=1)

    print("Encoding product texts to dense vectors...")
    embeddings = model.encode(df['embed_text'].tolist(), show_progress_bar=True, normalize_embeddings=True)

    supermarkets = df['supermarket'].unique()
    indices = {}
    id_maps = {}

    print("Building FAISS Indices per retailer...")
    for sm in supermarkets:
        sm_mask = df['supermarket'] == sm
        sm_idx = df[sm_mask].index.values
        sm_embeddings = embeddings[sm_mask]

        # Inner Product = Cosine Similarity since vectors are normalized
        index = faiss.IndexFlatIP(sm_embeddings.shape[1])
        index.add(np.asarray(sm_embeddings, dtype=np.float32))

        indices[sm] = index
        id_maps[sm] = sm_idx

    all_pairs = []

    print("Retrieving Cross-Retailer Candidates (Top-K)...")
    for sm_anchor in supermarkets:
        anchor_mask = df['supermarket'] == sm_anchor
        anchor_idx = df[anchor_mask].index.values
        anchor_embeddings = np.asarray(embeddings[anchor_mask], dtype=np.float32)

        for sm_target in supermarkets:
            if sm_anchor == sm_target:
                continue  # Cross-retailer only

            index = indices[sm_target]
            target_id_map = id_maps[sm_target]

            # Retrieve Top-K
            scores, target_indices_local = index.search(anchor_embeddings, TOP_K_CANDIDATES)

            for i, anchor_id in enumerate(anchor_idx):
                for j in range(TOP_K_CANDIDATES):
                    target_local_id = target_indices_local[i, j]
                    if target_local_id == -1:
                        continue  # FAISS empty padding

                    target_global_id = target_id_map[target_local_id]
                    score = float(scores[i, j])

                    # Store as (min, max) to deduplicate bidirectional A->B and B->A pairs later
                    pair = (min(int(anchor_id), int(target_global_id)), max(int(anchor_id), int(target_global_id)))
                    all_pairs.append({
                        'id_a': pair[0],
                        'id_b': pair[1],
                        'score': score,
                    })

    # Deduplicate
    pairs_df = pd.DataFrame(all_pairs).drop_duplicates(subset=['id_a', 'id_b'])
    print(f"Retrieved {len(pairs_df):,} unique candidate pairs.")
    return pairs_df
