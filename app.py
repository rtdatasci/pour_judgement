"""
Pour Judgement — drop a photo of your latte art, face the judge.

Gradio app for the Build Small Hackathon (Thousand Token Wood track).
Local-only inference: OpenCV metrics + a small VLM via llama.cpp.
"""

from __future__ import annotations

import gradio as gr

from judge import JUDGE_NAME, JUDGE_TITLE, judge
from scoring import score_image

# ----------------------------------------------------------------- UI pieces

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400..900;1,9..144,400..700&family=Inter:wght@400;600&display=swap');

:root, .gradio-container {
  --espresso: #241712;
  --roast: #3a2a20;
  --crema: #e9c9a0;
  --milk: #faf4eb;
  --ink: #2a1e18;
  --stamp: #8c3b2e;
}
.gradio-container {
  background: var(--espresso) !important;
  font-family: 'Inter', sans-serif !important;
  max-width: 860px !important;
  margin: 0 auto !important;
}
#title h1 {
  font-family: 'Fraunces', serif !important;
  font-weight: 900; font-size: 2.6rem; letter-spacing: -0.02em;
  color: var(--crema); text-align: center; margin-bottom: 0;
}
#subtitle p {
  color: #b59d85; text-align: center; font-size: 0.95rem; margin-top: 0.25rem;
}
#drop .image-container, #drop {
  border: 2px dashed #6b5443 !important; border-radius: 14px !important;
  background: var(--roast) !important;
}
#go {
  background: var(--crema) !important; color: var(--ink) !important;
  font-family: 'Fraunces', serif !important; font-weight: 700 !important;
  font-size: 1.15rem !important; border: none !important; border-radius: 10px !important;
}
#go:hover { background: #f3dcbb !important; }

/* The scorecard — signature element */
.scorecard {
  background: var(--milk); color: var(--ink); border-radius: 6px;
  padding: 28px 30px; position: relative;
  box-shadow: 0 12px 40px rgba(0,0,0,0.45);
  font-family: 'Inter', sans-serif;
}
.scorecard .header {
  font-family: 'Fraunces', serif; font-size: 0.8rem; letter-spacing: 0.18em;
  text-transform: uppercase; color: #7a6552;
  border-bottom: 1px solid #d9c8b4; padding-bottom: 10px; margin-bottom: 16px;
}
.scorecard .pattern {
  font-family: 'Fraunces', serif; font-style: italic; font-weight: 600;
  font-size: 1.5rem; line-height: 1.25; margin-bottom: 14px;
}
.scorecard .verdict { font-size: 1rem; line-height: 1.55; margin-bottom: 18px; }
.scorecard .tip {
  background: #f1e4d2; border-left: 4px solid var(--crema);
  padding: 12px 14px; border-radius: 0 8px 8px 0; font-size: 0.95rem;
}
.scorecard .tip b { font-family: 'Fraunces', serif; }
.stamp {
  position: absolute; top: 18px; right: 22px; transform: rotate(8deg);
  border: 3px solid var(--stamp); color: var(--stamp); border-radius: 8px;
  font-family: 'Fraunces', serif; font-weight: 900; font-size: 1.9rem;
  padding: 6px 14px; opacity: 0.85;
}
.bars { margin-top: 18px; }
.bar-row { display: flex; align-items: center; gap: 10px; margin: 7px 0; }
.bar-label { width: 92px; font-size: 0.78rem; letter-spacing: 0.06em;
  text-transform: uppercase; color: #7a6552; }
.bar-track { flex: 1; height: 10px; background: #e3d3bd; border-radius: 5px; overflow: hidden; }
.bar-fill { height: 100%; background: linear-gradient(90deg, #c89a64, #8c5a33); border-radius: 5px; }
.bar-num { width: 38px; text-align: right; font-size: 0.82rem; font-weight: 600; }
.judging { color: var(--crema); font-family: 'Fraunces', serif; font-style: italic;
  text-align: center; font-size: 1.05rem; }
footer { display: none !important; }
"""

JUDGING_LINES = "The judge lifts the cup to the light… swirls… inhales… deliberates…"


def render_card(result: dict, verdict: dict) -> str:
    s = result["subscores"]
    bars = "".join(
        f"""<div class="bar-row">
              <div class="bar-label">{name}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{val}%"></div></div>
              <div class="bar-num">{val:.0f}</div>
            </div>"""
        for name, val in s.items()
    )
    return f"""
    <div class="scorecard">
      <div class="stamp">{result['total']:.0f}</div>
      <div class="header">Official Scorecard — {JUDGE_NAME}, {JUDGE_TITLE}</div>
      <div class="pattern">“{verdict['pattern']}”</div>
      <div class="verdict">{verdict['verdict']}</div>
      <div class="tip"><b>The Tip:</b> {verdict['tip']}</div>
      <div class="bars">{bars}</div>
    </div>
    """


def face_the_judge(image_path, progress=gr.Progress()):
    if image_path is None:
        return "<div class='judging'>The judge cannot score an empty saucer. Add a photo.</div>"
    progress(0.1, desc="Measuring the pour…")
    result = score_image(image_path)
    progress(0.35, desc=JUDGING_LINES)
    verdict = judge(image_path, result)
    progress(1.0, desc="Verdict reached.")
    return render_card(result, verdict)


with gr.Blocks(title="Pour Judgement") as demo:
    gr.Markdown("# Pour Judgement", elem_id="title")
    gr.Markdown(
        f"Drop a photo of your latte art. *{JUDGE_NAME}* will take it from there.",
        elem_id="subtitle",
    )
    image = gr.Image(type="filepath", label=" ", elem_id="drop", height=340)
    go = gr.Button("Face the Judge", elem_id="go", size="lg")
    card = gr.HTML()

    go.click(face_the_judge, inputs=image, outputs=card)
    image.upload(lambda: "", outputs=card)  # clear old verdict on new photo

if __name__ == "__main__":
    demo.launch(css=CSS)
