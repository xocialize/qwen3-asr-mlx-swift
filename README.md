# qwen3-asr-mlx-swift

![platform](https://img.shields.io/badge/platform-macOS%2015%2B%20%C2%B7%20Apple%20Silicon-black)
![swift](https://img.shields.io/badge/Swift-6.2-orange)
![port code](https://img.shields.io/badge/port%20code-MIT-blue)

Swift-MLX port of the **Qwen3-ASR** architecture (`qwen3_asr`: AuT audio encoder + Qwen3 decoder)
and of the **Confucius4-R2T2 stable-prefix streaming loop** — NetEase Youdao's low-latency,
append-only real-time transcription fine-tune of `Qwen/Qwen3-ASR-1.7B`.

This is the engine-free CORE: mel front-end, encoder, decoder, one-shot transcription and the
streaming loop, with no MLXEngine dependency. The MLXEngine `stt` package that wraps it is
[`mlx-r2t2-stt-swift`](https://github.com/xocialize/mlx-r2t2-stt-swift).

- **Models:** [`netease-youdao/Confucius4-R2T2`](https://huggingface.co/netease-youdao/Confucius4-R2T2)
  (MLX tiers on `mlx-community/Confucius4-R2T2-{bf16,8bit}`); the stock `Qwen/Qwen3-ASR-{0.6B,1.7B}`
  load through the same code for one-shot transcription (they are not trained for the streaming loop).
- **Parity:** token-exact vs the Python-MLX rung on the fp32 CPU stream (G0–G4 in `PORTING-SPEC.md`);
  the Python rung itself is token-exact vs the PyTorch reference loop.
- **Licences:** port code MIT; weights by their publishers (see `NOTICE`).

## Install

```swift
.package(url: "https://github.com/xocialize/qwen3-asr-mlx-swift.git", from: "0.1.0"),
// …
.product(name: "Qwen3ASR", package: "qwen3-asr-mlx-swift"),
```

## Quick start

```swift
import Qwen3ASR

let dir = URL(fileURLWithPath: "/path/to/Confucius4-R2T2-8bit")   // config.json + tokenizer.json + *.safetensors
let model = try Qwen3ASRModel.load(directory: dir)
try await model.loadTokenizer(directory: dir)

// One-shot
let t = model.transcribe(samples: samples16k, language: "English")
print(t.text)

// Streaming — 160 ms chunks, committed text only, never revised
let stream = R2T2Stream(model: model, language: "English")
for chunk in microphoneChunks { for delta in stream.push(chunk) { print(delta, terminator: "") } }
print(stream.finish())
```

`R2T2Stream.Options` exposes the reference knobs (`chunkSeconds` 0.16, `lookaheadSeconds` 0.16,
`unfixedTokens` 1, the 16 s / 8 s rolling window, the `.server` / `.example` token schedule) and the
decode `strategy`: `.reencode` re-feeds the window every chunk like the vLLM reference; `.reuse`
(default) caches completed 8 s encoder blocks and their KV and re-prefills only what changed — same
text, less work.

## Gates

```bash
M=/path/to/Confucius4-R2T2-bf16      # the goldens were captured from the bf16 tier
swift run -c release qwen3asr-gates g0 --model $M                                  # key contract
swift run -c release qwen3asr-gates g1 --model $M --golden Tools/goldens/zh_test   # mel
swift run -c release qwen3asr-gates g2 --model $M --golden Tools/goldens/zh_test   # encoder
swift run -c release qwen3asr-gates g3 --model $M --golden Tools/goldens/zh_test   # LM logits
swift run -c release qwen3asr-gates g4 --model $M --golden Tools/goldens/ls_long   # the loop, token-exact
swift run -c release qwen3asr-gates stream --model $M --wav clip.wav --gpu --language English
```

The goldens and the clips they were captured from ship in `Tools/goldens/` (clip licences in
`Tools/goldens/FIXTURES.md`), so a clean clone runs every gate; only the weights are fetched. Goldens
come from `Tools/capture_goldens.py` — the Python-MLX rung on the fp32 CPU stream. Gates run on the
fp32 CPU stream by default; `--gpu` runs the stored dtype on Metal. `g4` on `ls_long` takes about
7.5 min in release (both strategies).

## Lineage

The encoder/decoder module layout follows Blaizzy/mlx-audio-swift (MIT); the streaming loop is a
port of `netease-youdao/Confucius4-R2T2` `r2t2/r2t2_asr.py` (Apache-2.0). See `NOTICE`.
