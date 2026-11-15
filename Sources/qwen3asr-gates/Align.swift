import ArgumentParser
import Foundation
import MLX
import Qwen3ASR

/// Forced-aligner parity: the Swift Qwen3-ForcedAligner against mlx-audio's Python implementation
/// (Tools/aligner_goldens.py): prompt ids identical, timestamp classes, final word times; plus warm latency.
struct Align: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "align", abstract: "Qwen3-ForcedAligner parity vs mlx-audio, and per-clip latency.")
    @Option(help: "Forced-aligner repo directory (config.json, *.safetensors, vocab.json, merges.txt).") var model: String
    @Option(help: "Goldens directory (clip*/ from Tools/aligner_goldens.py).") var golden: String = "Tools/goldens/aligner"
    @Flag(help: "Run on the GPU in the stored dtype (the goldens' setting) instead of fp32 on the CPU stream.") var gpu = false
    @Option(help: "Timed repeats per clip after one warm-up.") var repeats: Int = 5

    func run() async throws {
        if !gpu { MLX.Device.setDefault(device: Device(.cpu)) }
        print("# align — Qwen3-ForcedAligner parity\n")
        let t0 = Date()
        let aligner = try Qwen3ForcedAligner.load(directory: URL(fileURLWithPath: model), dtype: gpu ? nil : .float32)
        print("  loaded in \(String(format: "%.2f", Date().timeIntervalSince(t0))) s (\(gpu ? "gpu, stored dtype" : "cpu, fp32"))")
        let root = URL(fileURLWithPath: golden)
        let clips = try FileManager.default.contentsOfDirectory(atPath: root.path).filter { $0.hasPrefix("clip") }.sorted()
        var ok = true
        for clip in clips {
            let d = root.appendingPathComponent(clip)
            let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: d.appendingPathComponent("words.json"))) as! [String: Any]
            let text = meta["text"] as! String
            let g = try loadArrays(url: d.appendingPathComponent("golden.safetensors"))
            let samples = g["samples16k"]!.asArray(Float.self)
            let wantIDs = g["input_ids"]!.asArray(Int32.self).map(Int.init)
            let wantCls = g["timestamp_classes"]!.asArray(Int32.self).map(Int.init)
            let words = Qwen3ForcedAligner.wordList(text)
            ok = verdict(words == meta["word_list"] as! [String], "\(clip) word list", "\(words.count) words") && ok
            let (ids, cls) = aligner.timestampClasses(samples16k: samples, words: words)
            ok = verdict(ids == wantIDs, "\(clip) prompt ids", "\(ids.count) vs \(wantIDs.count)") && ok
            let flips = zip(cls, wantCls).filter { $0 != $1 }
            let worst = flips.map { abs($0 - $1) }.max() ?? 0
            ok = verdict(cls.count == wantCls.count && flips.count <= max(1, cls.count / 20) && worst <= 1, "\(clip) timestamp classes",
                         "\(flips.count)/\(cls.count) differ, worst by \(worst) class (80 ms)") && ok
            let got = aligner.align(samples16k: samples, text: text)
            let want = (meta["items"] as! [[String: Any]]).map { ($0["start"] as! Double, $0["end"] as! Double) }
            let dt = zip(got, want).map { max(abs($0.start - $1.0), abs($0.end - $1.1)) }.max() ?? .infinity
            ok = verdict(got.count == want.count && dt <= 0.0801, "\(clip) word times", String(format: "worst %.3f s", dt)) && ok
            var times: [Double] = []
            for _ in 0..<repeats { let s = Date(); _ = aligner.align(samples16k: samples, text: text); times.append(Date().timeIntervalSince(s)) }
            times.sort()
            print(String(format: "      latency %.1f s clip: median %.1f ms, min %.1f, max %.1f (%d runs, warm)", Double(samples.count) / 16000,
                         times[times.count / 2] * 1000, times.first! * 1000, times.last! * 1000, repeats))
        }
        print(ok ? "\nALL PASS" : "\nFAILURES")
        if !ok { throw ExitCode.failure }
    }
}

/// Scores a TTS run's word timings against this aligner, per take (one sentence chunk each): the manifest of
/// `zonos3-gates --timings` (name, text, words[{text, start, duration}]) with `<name>.wav` beside it. Prints the
/// onset / end error (median, p90) and the aligner's per-chunk latency (wav read + resample to 16 kHz excluded).
struct AlignScore: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "align-score", abstract: "Score TTS word timings against the forced aligner; per-chunk latency.")
    @Option(help: "Forced-aligner repo directory.") var model: String
    @Option(help: "Directory with manifest.json and <name>.wav takes.") var takes: String
    @Option(help: "Write the aligner's words per take here (JSON).") var out: String?

    func run() async throws {
        let aligner = try Qwen3ForcedAligner.load(directory: URL(fileURLWithPath: model))
        let dir = URL(fileURLWithPath: takes)
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("manifest.json"))) as! [[String: Any]]
        var onset: [Double] = [], end: [Double] = [], latency: [Double] = [], audio: [Double] = [], mismatched = 0
        var dump: [String: Any] = [:]
        for (k, take) in manifest.enumerated() {
            let name = take["name"] as! String, text = take["text"] as! String
            let words = take["words"] as! [[String: Any]]
            let samples = try WAV.load(dir.appendingPathComponent(name + ".wav").path)
            let t0 = Date()
            let ref = aligner.align(samples16k: samples, text: text)
            let dt = Date().timeIntervalSince(t0)
            if k > 0 { latency.append(dt); audio.append(Double(samples.count) / 16000) }  // the first call compiles kernels
            dump[name] = ref.map { ["text": $0.text, "start": $0.start, "end": $0.end] }
            guard ref.count == words.count else { mismatched += 1; continue }
            for (w, r) in zip(words, ref) {
                let s = w["start"] as! Double, d = w["duration"] as! Double
                onset.append(s - r.start); end.append(s + d - r.end)
            }
        }
        func stats(_ x: [Double]) -> String {
            let a = x.map { abs($0) * 1000 }.sorted()
            guard !a.isEmpty else { return "none" }
            return String(format: "median %.1f ms, p90 %.1f ms, mean signed %+.1f ms, worst %.1f ms (n %d)",
                          a[a.count / 2], a[Int(Double(a.count - 1) * 0.9)], x.reduce(0, +) / Double(x.count) * 1000, a.last!, a.count)
        }
        print("# align-score — \(manifest.count) takes, \(mismatched) with a different word count\n")
        print("  onset: \(stats(onset))")
        print("  end:   \(stats(end))")
        let l = latency.sorted()
        if !l.isEmpty {
            print(String(format: "  aligner latency per chunk (%.1f–%.1f s of audio): median %.1f ms, p90 %.1f ms, max %.1f ms (%d warm chunks)",
                         audio.min()!, audio.max()!, l[l.count / 2] * 1000, l[Int(Double(l.count - 1) * 0.9)] * 1000, l.last! * 1000, l.count))
        }
        if let out { try JSONSerialization.data(withJSONObject: dump, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: out)) }
    }
}
