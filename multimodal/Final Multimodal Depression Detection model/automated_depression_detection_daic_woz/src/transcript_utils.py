from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def clean_participant_id(participant_id) -> str:
    """Normalize Participant_ID values such as 303.0 -> 303."""
    raw = str(participant_id).strip()
    try:
        return str(int(float(raw)))
    except (TypeError, ValueError):
        return raw


def participant_to_transcript_path(participant_id, transcript_dir: str | Path) -> Path:
    pid = clean_participant_id(participant_id)
    transcript_dir = Path(transcript_dir)
    candidates = [
        transcript_dir / f"{pid}_TRANSCRIPT.csv",
        transcript_dir / f"{pid}_transcript.csv",
        transcript_dir / f"{pid}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _read_transcript_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # DAIC-WOZ transcript files are often tab-separated even if extension is .csv.
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t", engine="python")
    if df.shape[1] == 1:
        try:
            df = pd.read_csv(path, sep="\t", engine="python")
        except Exception:
            pass
    return df


def _pick_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)  # remove tags like <laughter>
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_participant_transcript(participant_id, transcript_dir: str | Path) -> str:
    """Load participant/user spoken transcript text for one DAIC-WOZ participant.

    It tries to remove Ellie/interviewer utterances and keep only participant answers.
    If speaker columns are missing, it falls back to all available text.
    """
    path = participant_to_transcript_path(participant_id, transcript_dir)
    if not path.exists():
        return ""

    try:
        df = _read_transcript_csv(path)
    except Exception:
        return ""

    text_col = _pick_column(
        df.columns,
        ["value", "transcript", "text", "sentence", "utterance", "content", "dialogue"],
    )
    if text_col is None:
        # fallback: use the last object-like column
        object_cols = [c for c in df.columns if df[c].dtype == object]
        text_col = object_cols[-1] if object_cols else df.columns[-1]

    speaker_col = _pick_column(df.columns, ["speaker", "role", "speaker_id"])
    if speaker_col is not None:
        speaker = df[speaker_col].astype(str).str.lower()
        participant_mask = speaker.str.contains("participant|client|subject|user", regex=True, na=False)
        ellie_mask = speaker.str.contains("ellie|interviewer|therapist|agent", regex=True, na=False)
        if participant_mask.any():
            df = df[participant_mask]
        elif ellie_mask.any():
            df = df[~ellie_mask]

    texts = [clean_text(x) for x in df[text_col].dropna().tolist()]
    texts = [t for t in texts if t]
    return " ".join(texts)
