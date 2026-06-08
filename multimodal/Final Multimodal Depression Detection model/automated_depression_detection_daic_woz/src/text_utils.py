from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

TEXT_EMBEDDING_DIM = 384
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def get_text_encoder(model_name: str = DEFAULT_TEXT_MODEL):
    if _sentence_transformers_available():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)

    if _transformers_available():
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        return tokenizer, model

    raise ImportError(
        "No text embedding library is installed. Install sentence-transformers or transformers to enable text branch inference."
    )


def _encode_with_transformers(tokenizer, model, texts: Iterable[str]) -> np.ndarray:
    import torch

    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * attention_mask).sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1e-9)
    embeddings = summed / counts
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings.cpu().numpy()


def encode_texts(texts: Iterable[str], model_name: str = DEFAULT_TEXT_MODEL) -> np.ndarray:
    texts = [str(t).strip() for t in texts]
    safe_texts = [t if t else "empty transcript" for t in texts]
    encoder = get_text_encoder(model_name)

    if isinstance(encoder, tuple):
        tokenizer, model = encoder
        emb = _encode_with_transformers(tokenizer, model, safe_texts)
    else:
        emb = encoder.encode(
            safe_texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    return np.asarray(emb, dtype=np.float32)


def zero_text_embedding() -> np.ndarray:
    return np.zeros((TEXT_EMBEDDING_DIM,), dtype=np.float32)
