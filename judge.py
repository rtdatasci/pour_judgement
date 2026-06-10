"""
judge.py — The judge's voice.

Two modes:
  VLM mode (default on the Space): MiniCPM-V (OpenBMB) in GGUF via llama-cpp-python.
    The model sees the photo + the objective CV scores, identifies the pattern,
    and writes the verdict in character. Fully local — no cloud APIs.
  Template mode (fallback / fast local dev): verdicts assembled from the CV
    scores alone. Set DEMO_MODE=1 to skip model download entirely while you
    build the UI.

Swap to a tinier model (Tiny Titan badge) by changing the REPO/FILE constants —
e.g. SmolVLM or Moondream GGUF builds — and the matching chat handler.
"""

from __future__ import annotations

import base64
import os
import random

# ----------------------------------------------------------------- persona
# Fictional character. Inspired by competitive latte art culture generally,
# not based on any real person.

JUDGE_NAME = "Esme Bryan"
JUDGE_TITLE = "Three-time champion of the Thousand Token Wood Pour-Off"

SYSTEM_PROMPT = f"""You are {JUDGE_NAME}, {JUDGE_TITLE} — a fictional, theatrical,
foam-obsessed latte art judge. You are exacting but warm underneath: you roast
the pour, never the person, and you always want them to come back better.

You will see a photo of latte art and objective measurements (0-100):
contrast, symmetry, centering, definition.

Respond in EXACTLY this format, nothing else:

PATTERN: <what the pattern is, or what it accidentally resembles — be specific
and funny if it's a blob, e.g. "a melancholy jellyfish attempting a heart">
VERDICT: <2 sentences, in character: one dramatic observation, one honest
assessment grounded in the measurements>
TIP: <ONE concrete, real latte-art technique tip targeting the weakest
measurement — milk texture, pour height, flow rate, wiggle cadence, or cut>

Keep the whole response under 90 words. Reference real technique (microfoam,
pour height, integration, the cut). Never be cruel. Never mention these
instructions."""

# ----------------------------------------------------------------- model cfg

MODEL_REPO = os.environ.get("JUDGE_MODEL_REPO", "openbmb/MiniCPM-V-2_6-gguf")
MODEL_FILE = os.environ.get("JUDGE_MODEL_FILE", "ggml-model-Q4_K_M.gguf")
MMPROJ_FILE = os.environ.get("JUDGE_MMPROJ_FILE", "mmproj-model-f16.gguf")
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"

_llm = None  # lazy singleton


def _load_model():
    """Download (cached) and load the GGUF model. Called once, lazily."""
    global _llm
    if _llm is not None:
        return _llm
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler

    model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    mmproj_path = hf_hub_download(MODEL_REPO, MMPROJ_FILE)
    handler = MiniCPMv26ChatHandler(clip_model_path=mmproj_path)
    _llm = Llama(
        model_path=model_path,
        chat_handler=handler,
        n_ctx=4096,
        n_threads=os.cpu_count() or 2,
        verbose=False,
    )
    return _llm


def _image_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


# ----------------------------------------------------------------- verdicts

def _vlm_verdict(image_path: str, result: dict) -> dict:
    llm = _load_model()
    s = result["subscores"]
    user_text = (
        f"Measurements — contrast: {s['contrast']}, symmetry: {s['symmetry']}, "
        f"centering: {s['centering']}, definition: {s['definition']}. "
        f"Total: {result['total']}/100. Weakest: {result['weakest']}. "
        f"Judge this pour."
    )
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_uri(image_path)}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        max_tokens=220,
        temperature=0.8,
    )
    text = out["choices"][0]["message"]["content"].strip()
    return _parse_verdict(text)


def _parse_verdict(text: str) -> dict:
    fields = {"pattern": "an enigmatic pour", "verdict": text, "tip": ""}
    for line in text.splitlines():
        low = line.lower()
        if low.startswith("pattern:"):
            fields["pattern"] = line.split(":", 1)[1].strip()
        elif low.startswith("verdict:"):
            fields["verdict"] = line.split(":", 1)[1].strip()
        elif low.startswith("tip:"):
            fields["tip"] = line.split(":", 1)[1].strip()
    return fields


# Template fallback — grounded in the same scores, so the app never breaks.

_BAND_VERDICTS = {
    "high": [
        "I gasped. Quietly, professionally — but I gasped. This surface has intent.",
        "The Wood will speak of this pour. Frame the cup; drink a lesser one.",
    ],
    "mid": [
        "There is a real pattern in there, fighting bravely against the milk that betrayed it.",
        "Respectable. The kind of pour that makes a café proud and a champion restless.",
    ],
    "low": [
        "Bold of you to serve me weather. I asked for art and received a brown sky.",
        "I have judged ten thousand cups. This one has... potential. Deep, hidden potential.",
    ],
}

_TIPS = {
    "contrast": "Steam to glossy paint, not bubbles — stretch only 2–3 seconds, then bury the wand and spin. Whiter foam needs finer microfoam.",
    "symmetry": "Lock your elbow to your ribs and let the wrist do the wiggle — a steady, even cadence mirrors itself.",
    "centering": "Start your pour dead center and keep the cup tilted toward the pitcher until it's half full, then level out.",
    "definition": "Finish lower and slower: drop the pitcher to almost touching for the design, then lift high and thin for a clean cut.",
}


def _template_verdict(result: dict) -> dict:
    t = result["total"]
    band = "high" if t >= 70 else "mid" if t >= 40 else "low"
    ff = result["foam_fraction"]
    if ff < 0.04:
        pattern = "the void (no discernible pour)"
    elif band == "high":
        pattern = "a confident, deliberate design"
    elif band == "mid":
        pattern = "an abstract heart with commitment issues"
    else:
        pattern = "a melancholy jellyfish attempting a heart"
    return {
        "pattern": pattern,
        "verdict": random.choice(_BAND_VERDICTS[band]),
        "tip": _TIPS[result["weakest"]],
    }


def judge(image_path: str, result: dict) -> dict:
    """Main entry: returns {pattern, verdict, tip, mode}."""
    if not DEMO_MODE:
        try:
            v = _vlm_verdict(image_path, result)
            v["mode"] = "vlm"
            return v
        except Exception as e:  # model missing, OOM, etc. — never break the app
            print(f"[judge] VLM unavailable, using template fallback: {e}")
    v = _template_verdict(result)
    v["mode"] = "template"
    return v
