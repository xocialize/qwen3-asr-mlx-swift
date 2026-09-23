// swift-tools-version: 6.2
// qwen3-asr-mlx-swift — the Qwen3-ASR family (Qwen/Qwen3-ASR-{0.6B,1.7B} and the fine-tunes that
// keep its architecture, first among them netease-youdao/Confucius4-R2T2) ported to Swift-MLX.
//
// Engine-free by design: this is the CORE — mel front-end, AuT audio encoder, Qwen3 decoder,
// greedy transcription and the R2T2 stable-prefix streaming loop. It depends on MLX,
// mlx-swift-lm's MLXLMCommon (KVCache, attention helpers) and swift-transformers (the Qwen
// byte-level BPE) only. The MLXEngine `stt` package that wraps it lives in
// mlx-r2t2-stt-swift; keeping the two apart is what lets the same core later serve the
// Apache-2.0 offline Qwen3-ASR tiers and the forced aligner.
//
// Parity discipline (PORTING-SPEC.md): the oracle stream is fp32 CPU. The mel front-end is
// computed in DOUBLE precision on Accelerate to reproduce transformers' float64 numpy pipeline,
// because the Python-MLX rung this port is gated against feeds numpy-derived features.
//
// Lineage: the encoder/decoder module layout follows Blaizzy/mlx-audio-swift's
// Sources/MLXAudioSTT/Models/Qwen3ASR (MIT, 2025 Prince Canuma), itself a port of
// Blaizzy/mlx-audio's qwen3_asr; see NOTICE.
import PackageDescription

let package = Package(
    name: "qwen3-asr-mlx-swift",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "Qwen3ASR", targets: ["Qwen3ASR"]),
        .executable(name: "qwen3asr-gates", targets: ["qwen3asr-gates"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.5"),
        // Major range, like the rest of the fleet: only long-stable MLXLMCommon pieces are used
        // (KVCache, attentionWithCacheUpdate), and a minor-only pin would block every consumer upgrade.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.4"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "Qwen3ASR",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Qwen3ASR",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "qwen3asr-gates",
            dependencies: [
                "Qwen3ASR",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/qwen3asr-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "Qwen3ASRTests",
            dependencies: ["Qwen3ASR"],
            path: "Tests/Qwen3ASRTests",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
