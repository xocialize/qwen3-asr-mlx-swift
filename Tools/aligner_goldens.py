"""Goldens for the Swift Qwen3-ForcedAligner (qwen3asr-gates align-parity): mlx-audio's Python implementation
(mlx_audio/stt/models/qwen3_asr/qwen3_forced_aligner.py) on a few clips.
usage: <mlx-audio venv>/bin/python Tools/aligner_goldens.py <repo or dir> <out dir> <wav> <text> [<wav> <text> ...]
Per clip: golden.safetensors (samples16k, input_ids, timestamp_classes = argmax at the <timestamp> rows) + words.json."""
import sys, json, pathlib, numpy as np, soundfile as sf, mlx.core as mx
from scipy.signal import resample_poly
from mlx_audio.stt import load
repo, out = sys.argv[1], pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
m = load(repo)
pairs = list(zip(sys.argv[3::2], sys.argv[4::2]))
for k, (wav, text) in enumerate(pairs):
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    if sr != 16000: x = resample_poly(x, 16000 // 100, sr // 100).astype(np.float32)
    feats, mask, n_audio = m._preprocess_audio(x)
    words, prompt = m.aligner_processor.encode_timestamp(text, "English")
    prompt = prompt.replace("<|audio_pad|>", "<|audio_pad|>" * n_audio)
    ids = m._tokenizer.encode(prompt, return_tensors="np", add_special_tokens=False)
    logits = m(mx.array(ids), input_features=feats, feature_attention_mask=mask); mx.eval(logits)
    cls = np.array(mx.argmax(logits, axis=-1))[0]
    ts = cls[ids[0] == m.config.timestamp_token_id]
    r = m.generate(x, text=text, language="English")
    d = out / f"clip{k}"; d.mkdir(exist_ok=True)
    mx.save_safetensors(str(d / "golden.safetensors"), {"samples16k": mx.array(x.astype(np.float32)), "input_ids": mx.array(ids[0].astype(np.int32)),
                                                       "timestamp_classes": mx.array(ts.astype(np.int32))})
    (d / "words.json").write_text(json.dumps({"text": text, "word_list": words, "items": [{"text": i.text, "start": i.start_time, "end": i.end_time} for i in r.items]}))
    print(d, len(ids[0]), n_audio, len(words), ts[:8])
