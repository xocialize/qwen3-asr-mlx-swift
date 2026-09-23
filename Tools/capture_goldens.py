#!/usr/bin/env python
"""Capture Swift-port goldens from the Python-MLX rung on the fp32 CPU stream.

usage: capture_goldens.py --model <mlx repo dir> --wav <16 kHz wav> --out <dir> [--language English] [--context ""]
Writes golden.safetensors (mel [T,128], features [N,2048], prompt_ids [L], logits_last [vocab], gen_ids [k]) + meta.json.
Needs the Python-MLX rung (`r2t2_session.py`) from the R2T2 eval harness — `r2t2-eval/` in
xocialize/mlxengine-audio-tools — and that harness's venv; pass its directory as --harness (or set
R2T2_HARNESS). The clip is copied beside the golden as `audio.<ext>` and named relatively, so the
golden directory is self-contained and a clean clone of this repo runs the gates.
"""
import argparse, json, os, shutil, sys
import numpy as np, soundfile as sf
import mlx.core as mx
mx.set_default_device(mx.cpu)   # before any mlx_audio import

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--wav", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--language", default=None); ap.add_argument("--context", default=""); ap.add_argument("--gen", type=int, default=8)
ap.add_argument("--loop", action="store_true")
ap.add_argument("--harness", default=os.environ.get("R2T2_HARNESS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "WIP", "r2t2-eval")),
    help="directory holding r2t2_session.py (default: $R2T2_HARNESS, else the original WIP layout)")
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.harness))
from r2t2_session import load_mlx_model, build_prompt_raw

os.makedirs(a.out, exist_ok=True)
fixture = "audio" + os.path.splitext(a.wav)[1]
shutil.copy2(a.wav, os.path.join(a.out, fixture))
model = load_mlx_model(a.model, dtype=mx.float32, cpu=True)
m = model._model
audio, sr = sf.read(a.wav, dtype="float32"); assert sr == 16000
if audio.ndim > 1: audio = audio.mean(axis=1)
feats_in, mask, n_audio = m._preprocess_audio(audio)
mel = np.array(feats_in)[0].T.astype(np.float32)          # [T, 128]
features = m.get_audio_features(feats_in, mask); mx.eval(features)
lang = m.config.support_languages and next((l for l in m.config.support_languages if a.language and l.lower() == a.language.lower()), a.language) or a.language
template = build_prompt_raw(a.context, lang)
prompt = template.replace("<|audio_pad|>", "<|audio_pad|>" * n_audio)
ids = np.asarray(m._tokenizer.encode(prompt, add_special_tokens=False), dtype=np.int32)
ids_mx = mx.array(ids)[None]
emb = m.model.embed_tokens(ids_mx)
pad = np.flatnonzero(ids == m.config.audio_token_id); s = int(pad[0])
embeds = mx.concatenate([emb[:, :s], features.astype(emb.dtype)[None], emb[:, s + n_audio:]], axis=1)
logits = m(ids_mx, input_embeddings=embeds, cache=None)[0, -1]; mx.eval(logits)
from mlx_audio.lm.generate import generate_step
gen = []
for tok, _ in generate_step(prompt=ids_mx[0], input_embeddings=embeds[0], model=m, max_tokens=a.gen):
    if tok in m._eos_token_ids(): break
    gen.append(int(tok))
mx.save_safetensors(os.path.join(a.out, "golden.safetensors"), {
    "mel": mx.array(mel), "features": features.astype(mx.float32), "prompt_ids": mx.array(ids),
    "logits_last": logits.astype(mx.float32), "gen_ids": mx.array(np.asarray(gen, dtype=np.int32))})
json.dump({"wav": fixture, "language": lang, "context": a.context, "audio_tokens": int(n_audio),
           "mel_frames": int(mel.shape[0]), "gen_text": m._tokenizer.decode(gen, skip_special_tokens=True),
           "model": os.path.abspath(a.model)}, open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False, indent=1)
print(f"golden: mel {mel.shape} features {features.shape} prompt {len(ids)} gen {gen} -> {a.out}")


# ---- loop goldens: python capture_goldens.py ... --loop  (writes loop.json next to golden.safetensors)
if "--loop" in sys.argv:
    from r2t2_session import make_mlx_streamer, run_streaming
    rec = []
    streamer, backend = make_mlx_streamer(model, record=rec)
    out = run_streaming(streamer, audio, chunk_ms=160, lookahead_ms=160, language=lang, mode="window", schedule="server")
    steps = [{"ids": r["ids"], "gen": s["gen"], "delta": None, "audio_s": s["audio_s"], "max_new_tokens": s["max_new_tokens"],
              "prefix": s["prefix"]} for r, s in zip(rec, out["steps"])]
    json.dump({"wav": fixture, "language": lang, "context": a.context, "text": out["text"],
               "detected_language": out["language"], "steps": steps,
               "chunks": [{"audio_end_s": c["audio_end_s"], "delta": c["delta"]} for c in out["chunks"]]},
              open(os.path.join(a.out, "loop.json"), "w"), ensure_ascii=False, indent=1)
    print(f"loop golden: {len(steps)} steps -> {out['text']!r}")
