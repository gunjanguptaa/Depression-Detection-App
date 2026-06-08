from pathlib import Path

from src.config import load_config
from src.transcript_utils import load_participant_transcript

cfg = load_config()
transcript_dir = Path(cfg["paths"]["transcript_dir"])

for pid in [303, 304, 305, 310]:
    text = load_participant_transcript(pid, transcript_dir)
    print(f"Participant {pid}: {len(text)} chars")
    print(text[:250].replace("\n", " "))
    print("-" * 60)
