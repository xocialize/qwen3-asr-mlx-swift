import ArgumentParser
import Foundation
import MLX
import Qwen3ASR

/// Parity gates for qwen3-asr-mlx-swift. Goldens come from Tools/capture_goldens.py (the
/// Python-MLX rung on the fp32 CPU stream); every gate here runs the Swift side on the SAME
/// stream unless --gpu is passed, because bf16-vs-fp32 flips greedy argmax on near-ties.
@main
struct Gates: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "qwen3asr-gates",
        abstract: "Parity gates + a transcribe lane for the Qwen3-ASR / Confucius4-R2T2 Swift port.",
        subcommands: [G0.self, G1.self, G2.self, G3.self, G4.self, Transcribe.self, Stream.self])
}

struct Common: ParsableArguments {
    @Option(help: "Model directory (config.json + tokenizer.json + *.safetensors).") var model: String
    @Flag(help: "Run on the GPU in the stored dtype instead of fp32 on the CPU stream.") var gpu = false

    func load() async throws -> Qwen3ASRModel {
        if !gpu { MLX.Device.setDefault(device: Device.cpu) }
        let t0 = Date()
        let m = try Qwen3ASRModel.load(directory: URL(fileURLWithPath: model), dtype: gpu ? nil : .float32)
        try await m.loadTokenizer(directory: URL(fileURLWithPath: model))
        print("  loaded \(URL(fileURLWithPath: model).lastPathComponent) in \(String(format: "%.1f", Date().timeIntervalSince(t0))) s"
              + " (\(gpu ? "gpu, stored dtype" : "cpu, fp32"))")
        return m
    }
}

func maxAbs(_ a: MLXArray, _ b: MLXArray) -> (maxAbs: Float, rel: Float) {
    let d = MLX.abs(a.asType(.float32) - b.asType(.float32))
    let m = d.max().item(Float.self)
    let scale = MLX.abs(b.asType(.float32)).max().item(Float.self)
    return (m, scale > 0 ? m / scale : m)
}

func verdict(_ ok: Bool, _ label: String, _ detail: String = "") -> Bool {
    print("  \(ok ? "PASS" : "FAIL")  \(label)\(detail.isEmpty ? "" : "  — \(detail)")")
    return ok
}

/// Goldens name their clip relative to their own directory (`Tools/goldens/<name>/audio.*`, shipped
/// in-repo so a clean clone runs every gate); an absolute path still works for ad-hoc captures.
func fixturePath(_ path: String, in dir: URL) -> String {
    path.hasPrefix("/") ? path : dir.appendingPathComponent(path).path
}

struct Golden {
    let dir: URL
    let arrays: [String: MLXArray]
    let meta: [String: Any]
    init(_ path: String) throws {
        dir = URL(fileURLWithPath: path)
        arrays = try loadArrays(url: dir.appendingPathComponent("golden.safetensors"))
        meta = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("meta.json"))) as? [String: Any] ?? [:]
    }
    var wav: String { fixturePath(meta["wav"] as? String ?? "", in: dir) }
    var language: String? { meta["language"] as? String }
    var context: String { meta["context"] as? String ?? "" }
}

struct G0: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "G0 — config, special ids, tokenizer, token-count contract.")
    @OptionGroup var common: Common
    func run() async throws {
        print("# G0 — key contract\n")
        let m = try await common.load()
        var ok = true
        let c = m.config
        ok = verdict(c.audioTokenId == 151676 && c.audioStartTokenId == 151669 && c.audioEndTokenId == 151670,
                     "audio special ids", "\(c.audioStartTokenId)/\(c.audioTokenId)/\(c.audioEndTokenId)") && ok
        ok = verdict(c.audioConfig.blockFrames == 800 && c.audioConfig.convChunkFrames == 100, "block geometry",
                     "chunk \(c.audioConfig.convChunkFrames) frames, block \(c.audioConfig.blockFrames)") && ok
        let tok = m.tokenizer!
        let asr = tok.encode(text: "<asr_text>", addSpecialTokens: false)
        ok = verdict(asr == [Qwen3ASRSpecialIDs.asrText], "<asr_text> is one special token", "\(asr)") && ok
        let pads = tok.encode(text: "<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|>", addSpecialTokens: false)
        ok = verdict(pads == [151669, 151676, 151676, 151670], "audio markers tokenize as singles", "\(pads)") && ok
        let ids = m.promptIDs(template: Qwen3ASRModel.promptTemplate(context: "", language: "Chinese"), audioTokens: 88)
        ok = verdict(ids.filter { $0 == 151676 }.count == 88 && ids.last == Qwen3ASRSpecialIDs.asrText,
                     "prompt: 88 pads + trailing <asr_text>", "\(ids.count) ids") && ok
        let n = m.parameters().flattened().count
        ok = verdict(n > 600, "parameter tree populated", "\(n) tensors") && ok
        ok = verdict(m.canonicalLanguage("chinese") == "Chinese" && m.canonicalLanguage("en") == "English", "language canonicalisation") && ok
        let rt = tok.decode(tokens: tok.encode(text: "之前有顾客自己带酒水，也没加收钱。 Hello, world!", addSpecialTokens: false), skipSpecialTokens: true)
        ok = verdict(rt == "之前有顾客自己带酒水，也没加收钱。 Hello, world!", "tokenizer round-trips zh/en", rt) && ok
        print("\n## G0 \(ok ? "PASSED" : "FAILED")")
        if !ok { throw ExitCode.failure }
    }
}

struct G1: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "G1 — Whisper log-mel vs the numpy golden.")
    @OptionGroup var common: Common
    @Option var golden: String
    func run() async throws {
        print("# G1 — mel front-end\n")
        let g = try Golden(golden)
        let m = try await common.load()
        let samples = try WAV.load(g.wav)
        let mel = m.melFeatures(samples)
        let ref = g.arrays["mel"]!
        var ok = verdict(mel.shape == ref.shape, "frame geometry", "\(mel.shape) vs \(ref.shape)")
        if ok {
            let (ma, rel) = maxAbs(mel, ref)
            let exact = MLX.all(mel .== ref).item(Bool.self)
            ok = verdict(ma <= 2e-6, "log-mel within 2e-6 of numpy float64 (complex64 bins)",
                         String(format: "max_abs %.3e rel %.3e%@", ma, rel, exact ? " — BIT-EXACT" : "")) && ok
        }
        print("\n## G1 \(ok ? "PASSED" : "FAILED")")
        if !ok { throw ExitCode.failure }
    }
}

struct G2: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "G2 — audio encoder features vs the fp32 CPU golden (whole clip + per block).")
    @OptionGroup var common: Common
    @Option var golden: String
    @Option(help: "Tolerance on max_abs (fp32 rounding across two MLX bindings; 0 = bit-exact).") var tol: Float = 1e-3
    func run() async throws {
        print("# G2 — encoder\n")
        let g = try Golden(golden)
        let m = try await common.load()
        let refMel = g.arrays["mel"]!
        let refFeat = g.arrays["features"]!
        // (a) from the GOLDEN mel, so this isolates the encoder from G1
        let feat = m.audioTower.encode(refMel)
        eval(feat)
        var ok = verdict(feat.shape == refFeat.shape, "token count", "\(feat.shape) vs \(refFeat.shape)")
        if ok {
            let (ma, rel) = maxAbs(feat, refFeat)
            ok = verdict(ma <= tol, "features from golden mel", String(format: "max_abs %.3e rel %.3e", ma, rel)) && ok
        }
        // (b) per-block encode equals whole-clip encode (the streaming cache's premise)
        let frames = refMel.dim(0)
        if frames > m.config.audioConfig.blockFrames {
            let b0 = m.audioTower.encodeBlock(refMel[0 ..< m.config.audioConfig.blockFrames])
            let whole = feat[0 ..< b0.dim(0)]
            let exact = MLX.all(b0 .== whole).item(Bool.self)
            ok = verdict(exact, "block 0 alone == block 0 inside the clip (blocks are independent)") && ok
        }
        // (c) end to end from the Swift mel
        let samples = try WAV.load(g.wav)
        let feat2 = m.audioTower.encode(m.melFeatures(samples))
        let (ma2, rel2) = maxAbs(feat2, refFeat)
        ok = verdict(ma2 <= tol, "features from the Swift mel", String(format: "max_abs %.3e rel %.3e", ma2, rel2)) && ok
        print("\n## G2 \(ok ? "PASSED" : "FAILED")")
        if !ok { throw ExitCode.failure }
    }
}

struct G3: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "G3 — LM prefill logits + first greedy tokens vs the fp32 CPU golden.")
    @OptionGroup var common: Common
    @Option var golden: String
    @Option var tol: Float = 5e-3
    func run() async throws {
        print("# G3 — language model\n")
        let g = try Golden(golden)
        let m = try await common.load()
        let ids = g.arrays["prompt_ids"]!.asArray(Int32.self).map(Int.init)
        let feats = g.arrays["features"]!
        let mine = m.promptIDs(template: Qwen3ASRModel.promptTemplate(context: g.context, language: g.language), audioTokens: feats.dim(0))
        var ok = verdict(mine == ids, "prompt ids identical to the Python rung", "\(mine.count) vs \(ids.count)")
        let emb = m.inputEmbeddings(ids: ids, audio: feats)
        let logits = m.logits(embeddings: emb, cache: nil)[0, -1]
        eval(logits)
        let ref = g.arrays["logits_last"]!
        let (ma, rel) = maxAbs(logits, ref)
        let top1 = argMax(logits).item(Int.self), rtop1 = argMax(ref).item(Int.self)
        ok = verdict(ma <= tol && top1 == rtop1, "last-position logits", String(format: "max_abs %.3e rel %.3e top1 %d vs %d", ma, rel, top1, rtop1)) && ok
        let gen = m.generate(embeddings: emb, cache: m.model.newCache(), maxNewTokens: 8)
        let rgen = g.arrays["gen_ids"]!.asArray(Int32.self).map(Int.init)
        ok = verdict(gen == rgen, "first greedy tokens token-exact", "\(gen) vs \(rgen)") && ok
        print("\n## G3 \(ok ? "PASSED" : "FAILED")")
        if !ok { throw ExitCode.failure }
    }
}

struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "One-shot transcription of a wav.")
    @OptionGroup var common: Common
    @Option var wav: String
    @Option var language: String?
    @Option var context: String = ""
    func run() async throws {
        let m = try await common.load()
        let samples = try WAV.load(wav)
        let t0 = Date()
        let t = m.transcribe(samples: samples, language: language, context: context)
        let dt = Date().timeIntervalSince(t0)
        print("  audio \(String(format: "%.2f", Double(samples.count) / 16000)) s · \(t.promptTokens) prompt · \(t.tokenIDs.count) new tokens · \(String(format: "%.2f", dt)) s")
        print("  language: \(t.language ?? "nil")")
        print("  text: \(t.text)")
    }
}
