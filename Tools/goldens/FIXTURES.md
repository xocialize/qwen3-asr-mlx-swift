# Gate fixtures

Each golden directory carries the clip it was captured from as `audio.*`, named relatively in its
`loop.json` / `meta.json`, so a clean clone runs every gate (`qwen3asr-gates g1–g4 --golden
Tools/goldens/<name>`). The goldens themselves come from `Tools/capture_goldens.py` — the
Python-MLX rung on the fp32 CPU stream, itself token-exact against the PyTorch reference loop.

| golden | clip | source | licence |
|---|---|---|---|
| `zh_test`, `zh_test_auto` | 6.74 s Mandarin (`zh_test_auto`: language auto-detected) | `resources/test.wav` in [netease-youdao/Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) | Apache License 2.0 |
| `ls_10s` | 10.07 s English | LibriSpeech test-clean `1188-133604-0041` | CC BY 4.0 |
| `ls_long` | 32.88 s English — crosses the 16 s window drop | LibriSpeech test-clean `7021-79730-0003` | CC BY 4.0 |
| `zh_mixed` | 7.98 s Mandarin with transliterated foreign names | FLEURS `cmn_hans_cn` test `149040922025982373` | CC BY 4.0 |

- LibriSpeech: V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based on
  public domain audio books", ICASSP 2015 — <https://www.openslr.org/12>.
- FLEURS: A. Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of
  Speech", 2022 — <https://huggingface.co/datasets/google/fleurs>.
