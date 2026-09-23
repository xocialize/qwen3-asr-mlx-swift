import AVFoundation
import Foundation

enum WAV {
    /// Mono float32 samples at `targetRate` (resampled by AVAudioConverter when needed).
    static func load(_ path: String, targetRate: Double = 16000) throws -> [Float] {
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
        let inFormat = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: frames) else { throw NSError(domain: "WAV", code: 1) }
        try file.read(into: inBuf)
        guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: targetRate, channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: inFormat, to: outFormat) else { throw NSError(domain: "WAV", code: 2) }
        let cap = AVAudioFrameCount(Double(frames) * targetRate / inFormat.sampleRate) + 1024
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: cap) else { throw NSError(domain: "WAV", code: 3) }
        var supplied = false
        var err: NSError?
        let status = conv.convert(to: outBuf, error: &err) { _, s in
            if supplied { s.pointee = .noDataNow; return nil }
            supplied = true; s.pointee = .haveData; return inBuf
        }
        if status == .error || err != nil { throw err ?? NSError(domain: "WAV", code: 4) }
        guard let ch = outBuf.floatChannelData?[0] else { throw NSError(domain: "WAV", code: 5) }
        return Array(UnsafeBufferPointer(start: ch, count: Int(outBuf.frameLength)))
    }
}
