---
title: Pour Judgement
emoji: ☕
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "6.17.3"
app_file: app.py
pinned: true
license: mit
tags:
  - track-thousand-token-wood
  - badge-off-the-grid
  - badge-llama-champion
  - badge-off-brand
  - badge-field-notes
  - openbmb
  - minicpm
  - llama-cpp
  - latte-art
  - computer-vision
---

# ☕ Pour Judgement

> *"Drop a photo of your latte art. Face the judge."*

**Demo video:** [ADD YOUTUBE LINK]
**Social post:** [ADD LINKEDIN/X LINK]

---

## What it does

You upload a photo of your latte art. **Esme Bryan** — fictional three-time
champion of the Thousand Token Wood Pour-Off — sizes it up and delivers a
verdict. She names what the pattern is (or what it accidentally resembles),
scores it across five dimensions, and gives you exactly one tip to fix on
your next pour.

No instructions needed. Drop a photo, press a button, face the judge.

---

## How it works

Everything runs locally — no cloud APIs, no external services.

**Step 1 — OpenCV scoring (instant, deterministic)**

Classical computer vision measures the pour before the model sees it:

- **Contrast** — tonal separation between milk foam and crema, gated by
  absolute foam lightness so muddy cups can't fake a high score
- **Symmetry** — mirror IoU of the foam pattern across up to five rotation
  angles, so tilted hearts aren't penalised
- **Centering** — how close the pattern centroid sits to the cup center
- **Definition** — edge sharpness along the foam/crema boundary
- **Milk Texture** — two signals inside the foam: Laplacian roughness
  (glossy microfoam is flat; bubbles are busy) and speckle ratio (how much
  of the raw segmentation mask is noise confetti). This is the bubbles-first
  metric: if texture is below 40, the milk is the problem and the tip
  addresses steaming before anything else.

A presence gate and a harshness curve (`score^1.35`) mean championship
scores must be genuinely earned.

**Step 2 — MiniCPM-V 2.6 (OpenBMB) via llama.cpp**

The photo — downscaled to 448px to cut vision-encoder passes on CPU — and
the objective sub-scores go into the model together. MiniCPM-V identifies
the pattern, writes the verdict in character, and targets the tip at the
weakest measurement. The model runs fully in-process via `llama-cpp-python`;
GGUF weights are cached on first boot.

If the model can't load for any reason, the app falls back to template
verdicts grounded in the same CV scores — it never breaks.

---

## Badges claimed

| Badge | Why |
|---|---|
| 🔌 Off the Grid | No cloud APIs — all inference runs in the Space |
| 🦙 Llama Champion | VLM served via `llama-cpp-python` |
| 🎨 Off-Brand | Custom scorecard UI — espresso/crema palette, Fraunces serif, stamped score card |
| 📓 Field Notes | [Blog post — ADD LINK] |

---

## Local development

```bash
git clone https://huggingface.co/spaces/build-small-hackathon/pour-judgement
cd pour-judgement
python -m venv .venv
source .venv/Scripts/activate   # Windows/Git Bash
pip install -r requirements.txt
DEMO_MODE=1 python app.py       # instant template verdicts, no model download
python app.py                   # full VLM mode (~5 GB download on first run)
```

### Swap to a tinier model (≤4B, Tiny Titan angle)

Set env vars — no code changes:

```bash
JUDGE_MODEL_REPO=openbmb/MiniCPM-V-2_6-gguf
JUDGE_MODEL_FILE=ggml-model-Q4_K_M.gguf     # swap for a smaller GGUF
JUDGE_MMPROJ_FILE=mmproj-model-f16.gguf
```

---

## Tech stack

- [Gradio](https://gradio.app) — UI
- [OpenCV](https://opencv.org) — cup detection, foam segmentation, five metrics
- [MiniCPM-V 2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6) (OpenBMB) — vision-language model
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — GGUF inference on CPU
- [Hugging Face Hub](https://huggingface.co) — model weight caching
