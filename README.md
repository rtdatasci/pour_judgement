---
title: Pour Judgement
emoji: ☕
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "6.17.3"
app_file: app.py
pinned: false
license: mit
---

# ☕ Pour Judgement

Drop a photo of your latte art. **Esme Bryan** — fictional three-time champion
of the Thousand Token Wood Pour-Off — measures it, names it, judges it, and
gives you exactly one tip to fix on your next pour.

Built for the **Build Small Hackathon** · Thousand Token Wood track.

## How it works (everything runs locally — no cloud APIs)

1. **OpenCV** finds the cup and computes objective metrics: contrast,
   symmetry, centering, definition → a 0–100 score.
2. **MiniCPM-V (OpenBMB), GGUF Q4, via llama.cpp** sees the photo + the
   measurements, identifies the pattern, and writes the verdict in character.
3. If the model can't load, the app falls back to template verdicts grounded
   in the same measurements — it never breaks.

## Local development

```bash
pip install -r requirements.txt
DEMO_MODE=1 python app.py     # instant template verdicts, no model download
python app.py                 # full VLM mode (downloads ~5 GB GGUF on first run)
```

## Swapping to a tinier model (Tiny Titan badge)

Set env vars — no code change needed:

```bash
JUDGE_MODEL_REPO=...  JUDGE_MODEL_FILE=...  JUDGE_MMPROJ_FILE=...
```

Note: a different model family may need a different llama-cpp chat handler
(see `judge.py::_load_model`).

## Badge checklist

- [x] Off the Grid — no cloud APIs, all inference in the Space
- [x] llama.cpp — VLM served through llama-cpp-python
- [x] Off-Brand — custom scorecard UI, no default Gradio look
- [ ] Field Notes — blog post (write before submitting!)
