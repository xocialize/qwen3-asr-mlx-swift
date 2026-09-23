# qwen3-asr-mlx-swift / mlx-r2t2-stt-swift — porting spec & gates (Confucius4-R2T2)

Source of truth for the model surface: the Python-MLX port in mlx-audio —
`mlx_audio/stt/models/qwen3_asr/{qwen3_asr.py, config.py}` (area fork `WIP/audio8-tts/mlx-audio` @
`1792021`) — and for the streaming protocol the reference `netease-youdao/Confucius4-R2T2`
`r2t2/r2t2_asr.py` + `example.py` + `ws_server.py` (master @ `c461192`, 2026-09-22), ported first to
Python-MLX as `WIP/r2t2-eval/r2t2_session.py` and gated there against the PyTorch reference (PR #3's
transformers backend, fp32 CPU). Same `mlx::core` underneath, so the Swift target is **token-exact
on the fp32 CPU stream**, not "close".

Checkpoint: `netease-youdao/Confucius4-R2T2` (2 038.1 M params, 4.08 GB bf16, one shard; NetEase
Youdao Model Use License Agreement — AB-L-0140). Task **AB-T-0167**; evaluation E20 in
`mlxengine-audio/Docs/ENHANCEMENTS.md`.

---

## Architecture (measured from the safetensors header)

| subtree | size | note |
|---|---|---|
| `thinker.audio_tower` (AuT) | 0.63 GB / 317.5 M | 24 × 1024, 16 heads, FFN 4096; 3 × Conv2d(k3,s2) → 8× in time, 128 → 16 mel; `conv_out` 7680→1024; sinusoidal positions per 100-frame chunk; attention bidirectional inside 800-frame (8 s) blocks, none across; `ln_post → proj1 → gelu → proj2` → 2048 |
| `thinker.model` (Qwen3) | 3.44 GB / 1 720.6 M | 28 × 2048, GQA 16/8, head 128, QK-RMSNorm, SwiGLU 6144, RoPE θ 1e6, tied vocab 151 936; the `mrope_section` in config is a no-op for ASR |
| `lm_head` | — | not materialised (tied) |

Frame geometry: Whisper 128-bin log-mel, n_fft 400 / hop 160 at 16 kHz → 100 frames/s; 13 audio
tokens per 100 frames → **12.5 tokens/s, 80 ms per token**. Prompt: `<|im_start|>system\n{context}
<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>\n<|im_start|>
assistant\n` + `language X<asr_text>` when forced. Special ids: audio_start 151669, audio_end
151670, audio_pad 151676, `<asr_text>` 151704, EOS {151645, 151643}.

### The streaming protocol (R2T2)
Per 160 ms chunk (first: 320 ms = chunk + lookahead): append to the audio window (≤ 16 s; once
exceeded, drop the oldest 8 s AND the text those 49/50 chunks committed); prompt = template +
committed text; greedy-decode ≤ `max_new_tokens` (adaptive: 2 per 160 ms, +0.5 while stalled on a
non-Chinese tail, ×2 when the last committed token is Chinese, cap 4); cut at the first `|`; roll
back 1 token (more if a UTF-8 character would split); commit the rest. `ws_server.py`'s
tail-repetition guard (≥ 5×) resets the state. Every step re-feeds the window from scratch on
vLLM; the Swift port's `.reuse` strategy caches completed 8 s encoder blocks whose mel did not
change and trims the KV cache back to head + those blocks, re-prefilling only the partial block,
the assistant header and the committed text. Blocks are attention-independent and the KV of a
position depends only on what precedes it, so the reuse is exact; the one coupling — Whisper's
global `max − 8` floor — is why a block is reused only when its mel compares EQUAL.

### Mel front-end precision
The Python rung feeds numpy float64 features (bins stored complex64, |X|² / mel / log10 in float64,
cast to float32 at the end). The Swift mel reproduces that pipeline in double precision on
Accelerate (`cblas_dgemm` DFT + mel, complex64 rounding of the bins) rather than a float32 MLX FFT,
because a ~1e-6 feature difference flips greedy argmax on near-ties and reads as a port defect
(the bf16 trap, mlx-vibevoice-asr-swift PORTING-SPEC).

---

## Phase table

Stamps are written AFTER each run, from tool output.

| Phase | Gate | Claim | Result |
|---|---|---|---|
| **V0** measurement spike | `WIP/r2t2-eval` (Python-MLX) | per-chunk latency, WER/CER by tier, Nemotron on the same audio, long-stream behaviour | in progress — see `WIP/r2t2-eval/MEASUREMENTS.md`. Early: **4-bit is unusable for streaming** (filler hallucinations on chunk 0, truncated tails, ~30 % WER vs ~1 % for bf16/8-bit); 8-bit tracks bf16 → shipping tier **int8** |
| **V1** Python-MLX rung | `compare_parity.py` (fp32 CPU) | `r2t2_session.py` token-exact per step vs the PyTorch reference loop (PR #3) | **fp32 CPU token-exact on every clip measured so far**: zh_test 73/73 (42 steps), zh_test_auto 158/158 (auto language), ls_10s 112/112 (62 steps); bf16 GPU exact on the zh clips, 109/112 on ls_10s with identical final text (the bf16 trap — near-tie flips); 8-bit GPU 68/75 · 158/158 · 108/112, final text identical in all. zh_mixed + ls_long (window drop) running |
| **G0** key contract | `qwen3asr-gates g0` | special ids, tokenizer round-trip, prompt shape | **PASSED 2026-09-22** — 707 tensors, 0 missing / 0 unused after the position table was un-registered (it is a constant, not a weight); ids 151669/151676/151670, `<asr_text>` 151704 single token; 88-pad prompt 106 ids; zh/en round-trip exact **Re-run 2026-09-23 (v0.1.0): PASSED.** |
| **G1** mel | `g1 --golden` | log-mel ≤ 2e-6 of numpy float64 | **PASSED 2026-09-22** — [674, 128] frames identical; max_abs **1.19e-7** (rel 8.6e-8) vs the numpy float64/complex64 pipeline **Re-run 2026-09-23 from the in-repo fixtures: PASSED** on `zh_test` and `ls_10s`. |
| **G2** encoder | `g2 --golden` | features from the golden mel and from the Swift mel; block-independence | **PASSED 2026-09-22** — from the golden mel **BIT-EXACT** (max_abs 0.0, 88 × 2048); from the Swift mel max_abs 9.8e-7 (rel 6.0e-6) **Re-run 2026-09-23 from the in-repo fixtures: PASSED** on `zh_test` and `ls_10s`. |
| **G3** language model | `g3 --golden` | last-position logits + first 8 greedy tokens vs the Python rung (fp32 CPU) | **PASSED 2026-09-22** — prompt ids identical (106); last-position logits **BIT-EXACT** (max_abs 0.0), top-1 101056; first 8 greedy tokens exact **Re-run 2026-09-23 from the in-repo fixtures: PASSED** on `zh_test` and `ls_10s`. |
| **G4** loop | `g4 --golden` (loop.json) | per-step ids token-exact vs the **Python-MLX** rung (same library — see *Cross-library near-ties*); `.reuse` == `.reencode` | **PASSED 2026-09-22 on both clips.** zh_test (42 steps, fp32 CPU): both strategies 73/73 tokens, transcript identical. **ls_long (205 steps, 32.88 s — CROSSES the 16 s window drop), release:** both strategies **385/385 tokens, 205/205 steps**, transcript identical (380 chars) — so the rolling window, the chunk-text trim and the cache invalidation across the drop are all exact. `.reuse` reused a mean of **85.6 KV positions per step (max 113)**. **Speed-up on an IDLE box: 95.2 s vs `.reencode`'s 156.4 s = 1.64×.** ⚠️ A first run of this gate reported 3.56× (98.6 vs 350.9 s); its `.reencode` half overlapped the Python latency matrix on the same GPU, which inflated the baseline. 1.64× is the number — and it agrees with the 6.7 s clip's 1.62×, so the gain is roughly constant rather than growing with window occupancy as first assumed. **Re-run 2026-09-23 (v0.1.0) from the in-repo fixtures — every golden now names its clip relatively and ships it in `Tools/goldens/` (licences in `FIXTURES.md`), so a clean clone runs this gate: PASSED on all five,** both strategies token-exact: zh_test 73/73, zh_test_auto 158/158, ls_10s 112/112, zh_mixed 94/94, ls_long 385/385 (205 steps, across the window drop). |
| **LAT** shipping-path latency | `qwen3asr-gates stream --gpu` (release) | 160 ms chunks hold realtime with margin in both headline languages | **PASSED 2026-09-22** — 8-bit cached: English steady median 65.4 ms (p99 124.8, RTF 0.440), Mandarin 66.7 ms (p99 122.3, RTF 0.441). bf16 Mandarin 88.3 ms with p99 169.1 — over budget in its tail — so **int8 is the tier on speed as well as size** |
| **G5** engine path | `r2t2-gates g5` | register / prepare / run / evict through MLXServeEngine | **PASSED 2026-09-22** (int8, real weights) — prepared in 24.9 s; transcript identical to the golden; 14 segments, starts non-decreasing; `speaker` nil throughout (matches `attributesSpeakers: false`); `detectedLanguage` Chinese; no `\|` marker leaked; evicted clean **Re-run 2026-09-23 (v0.1.0, quiet box): PASSED** — 6.7 s in 1.68 s (RTF 0.25), 14 segments, Chinese detected, evicted clean. |
| **G6** live session | `r2t2-gates g6` | `LiveTranscribing` / `STTSession` LIV-1..6, `.incremental`, final parity vs `run(STTRequest)` | **PASSED 2026-09-22** (int8, real weights, zh) — all of LIV-1..6: advertisement ⇔ conformance, 16 kHz / 30 s declared, **push 0.0001 s while the InferenceActor was held 0.50 s** (LIV-3), no chunk repeats its predecessor (LIV-2 incremental), index/isFinal/processedSeconds/committedThrough all sound with the watermark never ahead of consumed audio (LIV-4), **LIV-5 live transcript identical to `run(STTRequest)`**, cancel ends updates with no final chunk and the package still runs afterwards (LIV-6). **Re-run on English (ls_10s) also PASSED — LIV-5 identical at 27 WORDS**, which is the case a Chinese-only run cannot test: `.incremental` concatenates deltas verbatim, so a missing leading space would have fused the words (the bug the VibeVoice port hit). Byte-level BPE deltas carry their own leading space, so no package-side separator is needed here **Re-run 2026-09-23 (v0.1.0, quiet box) after the driver rewrite: PASSED on zh and en.** The driver used to BLOCK on a `DispatchSemaphore` inside `async` code whenever it waited for audio — one cooperative thread parked for the whole session, the only such wait in any audio package's library code. It now awaits an `AsyncStream<Void>` of wake-ups (newest-one buffering; push / finish / cancel yield, cancel finishes it) and takes the lock only through `withLock`, so the Swift 6 async-context warnings are gone. LIV-3 pushes 0.0000 s with the InferenceActor held 0.50 s; LIV-5 identical (zh; en 27 words); LIV-6 cancel semantics unchanged. |
| **KIT** offline conformance | `swift test` | manifest / CAN / MAT / weight-sourcing suites | **PASSED 2026-09-22 — 21/21** (10 manifest incl. the licence-layer and discipline declarations, 3 cancellation, 5 materialization, 3 weight sourcing) **Re-run 2026-09-23 (v0.1.0): 23/23** at engine 0.62.0 — adds `testFootprintCoversTheMeasuredPhysLine`, which fails if the declaration ever drops below the measured [VAL] line. |
| **LIC** licence plane | `swift test --filter ManifestTests` | the weight licence is engine-allowlisted, so `.permissiveOnly` admits with **no** `LicenseAdvisory` | **PASSED 2026-09-22** — `SPDXLicense.neteaseYoudaoModelUse` shipped in `mlx-engine-swift` **v0.57.0** (AB-A-0087; contract 1.42.0 unchanged). mlx-engine re-diffed the NetEase and bilibili agreement texts independently rather than taking AB-L-0140 on trust: **all of §3/§4/§5 byte-identical**, deltas confined to the title, §1.1–1.2/§1.4 names, §3.4(c)'s model name, the inserted §2.3 contact block and §6.2's venue (AB-R-0273). Consumer adopted the same day: package-local constant deleted, manifest on `.neteaseYoudaoModelUse`, engine floor 0.57.0, **22/22** incl. `testAdmitsUnderTheDefaultPermissiveOnlyPolicyWithNoAdvisory` |
| **VAL** footprint | `r2t2-gates val` | phys_footprint floor / activation vs the declaration (AB-L-0113) | **FAILED 2026-09-22** — int8 floor 2.69–2.70 GB vs 2.40 declared; activation **2.64 → 3.17 → 3.25 GB** at 6.7 / 163 / 182 s vs 1.50 declared; bf16 4.31 / 3.28 vs 4.20 / 1.50. Declared from the Python rung's allocator peak — the error AB-L-0113 names. Fix = ML[X] Audio Studio `Docs/R2T2-PLAN.md` M14-A (declare from phys, rounded up), then the plateau check (M14-C item 2) **PASSED 2026-09-23 (v0.1.0) at the corrected declaration — int8 2.80 GB resident + 3.60 GB activation, bf16 4.40 + 3.60.** Quiet box, sequential: int8 floor 2.69 / 2.70 / 2.70 GB, activation **2.62 / 3.10 / 3.10 GB** at 6.7 / 163 / 182 s (a plateau from 163 s on); bf16 floor 4.31, activation 3.24 GB at 182 s. Highest reading on any run, including runs taken under load (which read 0.1–0.4 GB low or high — AB-L-0118): 3.28 GB, so 3.60 leaves ~10 %. What the activation is (`r2t2-gates mem`, AB-L-0155): the model's own MLX working set is ~1.1 GB above the weights and flat in clip length; the rest is the engine's recycling pool at its 2 GiB automatic cap (every chunk re-prefills a new-length prompt); ~2 GB stays mapped in the Metal heap after `clearCache` (AB-L-0081), with MLX active back at the weights exactly — no leak. With pooling off, phys activation is ~0.06 GB but runs are ~50 % slower; the pool is engine policy, so the declaration is pool-inclusive. |
