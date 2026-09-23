import ArgumentParser
import Foundation
import MLX
import Qwen3ASR

/// Shared driver: pushes a wav through R2T2Stream in `pushSamples` slices (100 ms by default, the
/// way a microphone would), then finishes.
func runLoop(model: Qwen3ASRModel, samples: [Float], language: String?, context: String,
             strategy: R2T2Stream.Strategy, pushSamples: Int = 1600, onDelta: ((String) -> Void)? = nil) -> R2T2Stream {
    var opts = R2T2Stream.Options()
    opts.strategy = strategy
    let s = R2T2Stream(model: model, language: language, context: context, options: opts)
    var pos = 0
    while pos < samples.count {
        let end = min(pos + pushSamples, samples.count)
        for d in s.push(Array(samples[pos ..< end])) { onDelta?(d) }
        pos = end
    }
    let tail = s.finish()
    if !tail.isEmpty { onDelta?(tail) }
    return s
}

struct G4: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "G4 — streaming loop token-exact vs the Python-MLX rung (fp32 CPU), and .reuse == .reencode.")
    @OptionGroup var common: Common
    @Option(help: "Golden directory holding loop.json (capture_goldens.py --loop).") var golden: String
    func run() async throws {
        print("# G4 — stable-prefix loop\n")
        let dir = URL(fileURLWithPath: golden)
        let loop = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("loop.json"))) as! [String: Any]
        let wav = fixturePath(loop["wav"] as! String, in: dir)
        let language = loop["language"] as? String
        let context = loop["context"] as? String ?? ""
        let refSteps = loop["steps"] as! [[String: Any]]
        let refText = loop["text"] as! String
        let m = try await common.load()
        let samples = try WAV.load(wav)
        print("  audio \(String(format: "%.2f", Double(samples.count) / 16000)) s · \(refSteps.count) golden steps\n")
        var ok = true
        for strategy in [R2T2Stream.Strategy.reencode, .reuse] {
            let t0 = Date()
            let s = runLoop(model: m, samples: samples, language: language, context: context, strategy: strategy)
            let dt = Date().timeIntervalSince(t0)
            var firstDiv: Int? = nil
            var matched = 0, total = 0
            for (i, ref) in refSteps.enumerated() {
                let rids = (ref["ids"] as! [Int])
                guard i < s.steps.count else { if firstDiv == nil { firstDiv = i }; break }
                let mine = s.steps[i].generatedIDs
                total += max(rids.count, mine.count)
                matched += zip(rids, mine).filter { $0 == $1 }.count
                if rids != mine && firstDiv == nil { firstDiv = i }
            }
            let exact = firstDiv == nil && s.steps.count == refSteps.count
            ok = verdict(exact, "[\(strategy)] per-step ids token-exact vs the Python rung",
                         "\(matched)/\(total) tokens · \(s.steps.count) vs \(refSteps.count) steps · \(String(format: "%.1f", dt)) s") && ok
            if let d = firstDiv, d < s.steps.count, d < refSteps.count {
                print("        first divergence at step \(d) (\(String(format: "%.2f", s.steps[d].audioEndSeconds)) s):")
                print("          python: \(refSteps[d]["gen"] as? String ?? "")  \(refSteps[d]["ids"] as! [Int])")
                print("          swift : \(s.steps[d].generated)  \(s.steps[d].generatedIDs)")
            }
            ok = verdict(s.committedText == refText, "[\(strategy)] committed transcript identical",
                         s.committedText == refText ? "\(s.committedText.count) chars" : "\n        swift : \(s.committedText)\n        python: \(refText)") && ok
            if strategy == .reuse {
                let reused = s.steps.map(\.reusedTokens)
                print("        reuse: KV positions reused per step — max \(reused.max() ?? 0), mean \(String(format: "%.1f", Double(reused.reduce(0, +)) / Double(max(1, reused.count))))")
            }
        }
        print("\n## G4 \(ok ? "PASSED" : "FAILED")")
        if !ok { throw ExitCode.failure }
    }
}

struct Stream: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Stream a wav through the R2T2 loop and report per-chunk latency.")
    @OptionGroup var common: Common
    @Option var wav: String
    @Option var language: String?
    @Option var context: String = ""
    @Option(help: "reencode | reuse") var strategy: String = "reuse"
    @Flag(help: "Print each committed delta as it lands.") var verbose = false
    func run() async throws {
        let m = try await common.load()
        let samples = try WAV.load(wav)
        let strat: R2T2Stream.Strategy = strategy == "reencode" ? .reencode : .reuse
        // warm-up on the first 4 s so kernels are compiled before we time
        _ = runLoop(model: m, samples: Array(samples.prefix(4 * 16000)), language: language, context: context, strategy: strat)
        MLX.GPU.resetPeakMemory()
        let t0 = Date()
        let s = runLoop(model: m, samples: samples, language: language, context: context, strategy: strat) { d in
            if verbose { print("  + \(d)") }
        }
        let wall = Date().timeIntervalSince(t0)
        let secs = Double(samples.count) / 16000
        let walls = s.steps.dropLast().map { $0.wallSeconds * 1000 }.sorted()
        let steady = s.steps.filter { $0.audioEndSeconds >= 16 }.map { $0.wallSeconds * 1000 }.sorted()
        func pct(_ a: [Double], _ p: Double) -> Double { a.isEmpty ? .nan : a[min(a.count - 1, Int(Double(a.count) * p))] }
        print("\n  strategy \(strategy) · audio \(String(format: "%.1f", secs)) s · \(s.steps.count) steps · wall \(String(format: "%.1f", wall)) s · rtf \(String(format: "%.3f", wall / secs))")
        print("  chunk ms: median \(String(format: "%.1f", pct(walls, 0.5))) · p90 \(String(format: "%.1f", pct(walls, 0.9))) · p99 \(String(format: "%.1f", pct(walls, 0.99))) · max \(String(format: "%.1f", walls.last ?? .nan))")
        if !steady.isEmpty {
            print("  steady state (window full, ≥16 s): median \(String(format: "%.1f", pct(steady, 0.5))) · p90 \(String(format: "%.1f", pct(steady, 0.9))) ms")
        }
        let reused = s.steps.map(\.reusedTokens)
        print("  prompt tokens (last step) \(s.steps.last?.promptTokens ?? 0) · KV reused mean \(String(format: "%.1f", Double(reused.reduce(0, +)) / Double(max(1, reused.count))))")
        print("  hallucination resets \(s.hallucinationResets) · peak \(String(format: "%.2f", Double(MLX.GPU.peakMemory) / 1e9)) GB")
        print("  language: \(s.language)")
        print("  text: \(s.committedText)")
    }
}
