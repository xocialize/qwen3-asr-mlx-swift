import ArgumentParser
import Darwin
import Foundation
import MLX
import Qwen3ASR

/// Diagnostic, not a gate: does a long R2T2 session's memory grow with its length, and if so, whose
/// is it? Streams a wav through `R2T2Stream` in 100 ms pushes on the GPU in the stored dtype, with
/// MLX's pool capped the way MLXEngine's automatic policy caps it. Every `--every` seconds of audio,
/// BETWEEN steps (so no step's temporaries are alive), it prints phys_footprint beside:
///   active     bytes referenced by live MLX arrays (weights, KV cache, cached encoder blocks)
///   pool       MLX's recycling cache, bounded by the cap
///   heap       the malloc zones in use (Swift arrays and strings: the step log, the committed text)
///   other      phys − active − pool − heap: memory no counter owns (Metal heaps, driver state)
/// plus MLX's high-water mark since the previous line. Growth in `active` is a reference held too
/// long, in `heap` a Swift collection that only appends, in `other` retention below MLX.
struct Creep: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Memory over session length: phys vs MLX active / pool vs the malloc heap, sampled between steps.")
    @Option(help: "Model directory (config.json + tokenizer.json + *.safetensors).") var model: String
    @Option var wav: String
    @Option var language: String?
    @Option(help: "Seconds of audio between samples.") var every: Double = 60
    @Option(help: "MLX pool cap in bytes; MLXEngine's automatic policy is min(2 GiB, 5 % of the budget).")
    var pool: Int = 2 << 30

    func run() async throws {
        let m = try Qwen3ASRModel.load(directory: URL(fileURLWithPath: model), dtype: nil)
        try await m.loadTokenizer(directory: URL(fileURLWithPath: model))
        Memory.cacheLimit = pool
        let samples = try WAV.load(wav)
        let s = R2T2Stream(model: m, language: language)
        let t0 = Date()
        func gb(_ b: Int) -> String { String(format: "%6.3f", Double(b) / 1e9) }
        func line(_ label: String) {
            let phys = physFootprint(), active = Memory.activeMemory, cache = Memory.cacheMemory, heap = mallocInUse()
            print("  \(label.padding(toLength: 16, withPad: " ", startingAt: 0)) phys \(gb(phys))  active \(gb(active))  "
                  + "pool \(gb(cache))  heap \(gb(heap))  other \(gb(phys - active - cache - heap))  "
                  + "mlx-peak \(gb(Memory.peakMemory))  steps \(s.steps.count)  text \(s.committedText.utf8.count) B  "
                  + "wall \(String(format: "%.0f", Date().timeIntervalSince(t0))) s")
            fflush(stdout)
            Memory.peakMemory = 0   // the setter resets the high-water mark
        }
        print("# CREEP — \(URL(fileURLWithPath: model).lastPathComponent) · \(URL(fileURLWithPath: wav).lastPathComponent)"
              + " · \(String(format: "%.1f", Double(samples.count) / 16000 / 60)) min · pool cap \(pool) B\n")
        line("loaded")
        var pos = 0, next = every
        while pos < samples.count {
            let end = min(pos + 1600, samples.count)
            _ = s.push(Array(samples[pos ..< end]))
            pos = end
            if Double(pos) / 16000 >= next {
                line(String(format: "%.1f min", Double(pos) / 16000 / 60))
                next += every
            }
        }
        _ = s.finish()
        line("finished")
        Memory.clearCache()
        line("+ clearCache")
    }
}

/// The number Activity Monitor and the jetsam limit use — resident, compressed and GPU memory owned
/// by this process.
func physFootprint() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Int(info.phys_footprint) : -1
}

/// Bytes allocated from every malloc zone and not yet freed.
func mallocInUse() -> Int {
    var stats = malloc_statistics_t()
    malloc_zone_statistics(nil, &stats)
    return Int(stats.size_in_use)
}
