from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .data_utils import build_examples_from_split
from .text_utils import TEXT_EMBEDDING_DIM, encode_texts, zero_text_embedding
from .transcript_utils import clean_participant_id, load_participant_transcript


def build_text_embedding_cache(
    participant_ids: list[str],
    transcript_dir: str | Path,
    cache_path: str | Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    overwrite_cache: bool = False,
) -> Dict[str, np.ndarray]:
    """Create/load participant_id -> sentence embedding cache."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not overwrite_cache:
        with cache_path.open("rb") as f:
            return pickle.load(f)

    unique_ids = sorted({clean_participant_id(pid) for pid in participant_ids})
    texts = [load_participant_transcript(pid, transcript_dir) for pid in unique_ids]

    non_empty = sum(1 for t in texts if t.strip())
    print(f"Transcript texts found for {non_empty}/{len(unique_ids)} participants")

    if non_empty == 0:
        mapping = {pid: zero_text_embedding() for pid in unique_ids}
    else:
        embeddings = encode_texts(texts, model_name=model_name)
        mapping = {pid: emb.astype(np.float32) for pid, emb in zip(unique_ids, embeddings)}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(mapping, f)
    return mapping


def build_multimodal_examples_from_split(
    split_csv: str | Path,
    audio_dir: str | Path,
    transcript_dir: str | Path,
    cfg_features: dict,
    cache_dir: str | Path,
    split_name: str,
    overwrite_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return X_audio, X_text, y, metadata.

    Audio examples are segment-level. The same participant transcript embedding is repeated for each audio segment.
    """
    cache_dir = Path(cache_dir)
    X_audio, y, metadata = build_examples_from_split(
        split_csv,
        audio_dir,
        cfg_features,
        cache_path=cache_dir / f"{split_name}_features.npz",
        overwrite_cache=overwrite_cache,
    )

    participant_ids = [clean_participant_id(x) for x in metadata["participant_id"].tolist()]
    text_cache = build_text_embedding_cache(
        participant_ids,
        transcript_dir=transcript_dir,
        cache_path=cache_dir / f"{split_name}_text_embeddings.pkl",
        overwrite_cache=overwrite_cache,
    )

    X_text = []
    for pid in participant_ids:
        X_text.append(text_cache.get(pid, zero_text_embedding()))
    X_text = np.vstack(X_text).astype(np.float32)

    if X_text.shape[1] != TEXT_EMBEDDING_DIM:
        raise ValueError(f"Expected text embedding dim {TEXT_EMBEDDING_DIM}, got {X_text.shape[1]}")

    metadata = metadata.copy()
    metadata["participant_id_clean"] = participant_ids
    return X_audio, X_text, y, metadata
