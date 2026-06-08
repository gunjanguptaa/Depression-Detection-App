# Multimodal Upgrade Patch

Copy these files into your existing project root, keeping the same folder structure.

Then run:

```powershell
python scripts\quick_multimodal_check.py
python -m src.train_multimodal
python -m src.evaluate_multimodal
streamlit run app_multimodal.py
```

Extra dependencies:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers transformers faster-whisper
```

Use `app_multimodal.py` instead of replacing your old `app.py`, so your audio-only app remains safe.
