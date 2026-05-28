"""Level A: Candidate Generation via Vector Search."""

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, TOP_K_CANDIDATES


def create_embedding_text(row) -> str:
    """Build embedding text from normalized_name (SM-prefix-free, cleaned) + unit_type.

    Using normalized_name rather than the raw name gives FAISS comparable tokens
    across supermarkets: SM prefixes ('ASDA', 'Sainsbury's', 'Tesco') are already
    stripped by the normalization pipeline, and brand/descriptor terms are
    standardised.  Appending unit_type ('g' / 'ml') helps FAISS avoid matching
    weight products against volume products.
    """
    norm_name = str(row.get('normalized_name', '')).strip()
    norm_name = '' if norm_name == 'nan' else norm_name
    # Strip trailing ellipsis from source-truncated Morrisons names (U+2026 '…').
    # The truncation character adds noise to the embedding and misaligns the vector
    # from the equivalent full-name embedding at other supermarkets.
    norm_name = norm_name.rstrip('\u2026').strip()

    brand = str(row.get('known_brand_clean', '')).lower().strip()
    brand = '' if brand in ('nan', 'none', '') else brand

    unit_type = str(row.get('unit_type', '')).strip()
    unit_type = '' if unit_type in ('nan', 'none', '') else unit_type

    # Combine: brand gives a strong anchor when present; unit_type discriminates
    # g-products from ml-products that share the same name tokens.
    parts = [p for p in [brand, norm_name, unit_type] if p]
    return ' '.join(parts)




def _collect_pairs(anchor_idx: np.ndarray, scores: np.ndarray, target_indices_local: np.ndarray,
                   target_id_map: np.ndarray, k: int) -> list[dict]:
    pairs = []
    for i, anchor_id in enumerate(anchor_idx):
        for j in range(k):
            target_local_id = target_indices_local[i, j]
            if target_local_id == -1:
                continue
            target_global_id = target_id_map[target_local_id]
            score = float(scores[i, j])
            pair = (min(int(anchor_id), int(target_global_id)), max(int(anchor_id), int(target_global_id)))
            pairs.append({'id_a': pair[0], 'id_b': pair[1], 'score': score})
    return pairs


def retrieve_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict, dict]:
    """Builds FAISS indices per retailer and retrieves cross-retailer Top-K candidates.

    Returns
    -------
    pairs_df : DataFrame with columns id_a, id_b, score
    embeddings : (N, D) float32 array — one row per product, same order as df.index
    indices : dict supermarket → faiss.IndexFlatIP
    id_maps : dict supermarket → global product-index array
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    df = df.copy()
    df['embed_text'] = df.apply(create_embedding_text, axis=1)

    print("Encoding product texts to dense vectors...")
    embeddings = model.encode(df['embed_text'].tolist(), show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    supermarkets = df['supermarket'].unique()
    indices: dict = {}
    id_maps: dict = {}

    print("Building FAISS Indices per retailer...")
    for sm in supermarkets:
        sm_mask = df['supermarket'] == sm
        sm_idx = df[sm_mask].index.values
        sm_embeddings = embeddings[sm_mask]

        # Inner Product = Cosine Similarity since vectors are normalized
        index = faiss.IndexFlatIP(sm_embeddings.shape[1])
        index.add(sm_embeddings)

        indices[sm] = index
        id_maps[sm] = sm_idx

    all_pairs: list[dict] = []

    print("Retrieving Cross-Retailer Candidates (Top-K)...")
    for sm_anchor in supermarkets:
        anchor_mask = df['supermarket'] == sm_anchor
        anchor_idx = df[anchor_mask].index.values
        anchor_embeddings = embeddings[anchor_mask]

        for sm_target in supermarkets:
            if sm_anchor == sm_target:
                continue  # Cross-retailer only

            scores, target_indices_local = indices[sm_target].search(anchor_embeddings, TOP_K_CANDIDATES)
            all_pairs.extend(_collect_pairs(anchor_idx, scores, target_indices_local, id_maps[sm_target], TOP_K_CANDIDATES))

    pairs_df = pd.DataFrame(all_pairs).drop_duplicates(subset=['id_a', 'id_b'])
    print(f"Retrieved {len(pairs_df):,} unique candidate pairs.")
    return pairs_df, embeddings, indices, id_maps
