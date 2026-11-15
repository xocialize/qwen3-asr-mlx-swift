// ForcedAligner.swift — Qwen3-ForcedAligner (Qwen/Qwen3-ForcedAligner-0.6B, Apache-2.0): word timestamps for a clip
// whose transcript is known, in ONE non-autoregressive forward. The same AuT encoder and Qwen3 decoder as Qwen3-ASR;
// the head classifies every `<timestamp>` slot of the prompt into one of `classify_num` 80 ms bins.
// A port of Blaizzy/mlx-audio mlx_audio/stt/models/qwen3_asr/qwen3_forced_aligner.py (MIT, 2025 Prince Canuma):
// ForceAlignProcessor (word lists, the longest-non-decreasing timestamp repair), the prompt, the forward. Japanese and
// Korean word segmentation (nagisa / soynlp upstream) are not ported: those texts are split on whitespace and CJK
// ideographs only.
import AVFoundation
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// `config.json` of a forced-aligner repo: the Qwen3-ASR layout plus the timestamp head.
public struct Qwen3ForcedAlignerConfig: Decodable, Sendable {
    public var base: Qwen3ASRConfig
    public var classifyNum: Int = 5000
    public var timestampTokenId: Int = 151705
    /// Milliseconds per timestamp class.
    public var timestampSegmentTime: Double = 80

    enum Keys: String, CodingKey {
        case thinkerConfig = "thinker_config", classifyNum = "classify_num"
        case timestampTokenId = "timestamp_token_id", timestampSegmentTime = "timestamp_segment_time"
    }

    public init(from decoder: Decoder) throws {
        base = try Qwen3ASRConfig(from: decoder)
        let c = try decoder.container(keyedBy: Keys.self)
        let t = try? c.nestedContainer(keyedBy: Keys.self, forKey: .thinkerConfig)
        func value<T: Decodable>(_ type: T.Type, _ key: Keys) throws -> T? {
            if let t, let v = try t.decodeIfPresent(type, forKey: key) { return v }
            return try c.decodeIfPresent(type, forKey: key)
        }
        classifyNum = try value(Int.self, .classifyNum) ?? 5000
        timestampTokenId = try value(Int.self, .timestampTokenId) ?? 151705
        timestampSegmentTime = try value(Double.self, .timestampSegmentTime) ?? 80
    }
}

/// One aligned unit (a word, or one CJK character) in seconds from the start of the clip.
public struct AlignedWord: Sendable, Equatable, Codable {
    public let text: String
    public let start: Double
    public let end: Double
    public var duration: Double { end - start }
}

public final class Qwen3ForcedAligner: Module {
    public let config: Qwen3ForcedAlignerConfig
    @ModuleInfo(key: "audio_tower") public var audioTower: Qwen3ASRAudioEncoder
    @ModuleInfo(key: "model") public var model: Qwen3ASRTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    public let mel = WhisperLogMel()
    public private(set) var wordTokenizer: Qwen2WordTokenizer?

    public init(_ cfg: Qwen3ForcedAlignerConfig) {
        config = cfg
        _audioTower.wrappedValue = Qwen3ASRAudioEncoder(cfg.base.audioConfig)
        _model.wrappedValue = Qwen3ASRTextModel(cfg.base.textConfig)
        _lmHead.wrappedValue = Linear(cfg.base.textConfig.hiddenSize, cfg.classifyNum, bias: false)
        super.init()
    }

    /// Loads a repo directory (`config.json`, `*.safetensors`, `vocab.json` + `merges.txt`). Quantized mlx-community
    /// repos rebuild exactly the modules that carry `.scales` (the decoder); the audio tower stays as stored.
    public static func load(directory: URL, dtype: DType? = nil) throws -> Qwen3ForcedAligner {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else { throw Qwen3ASRError.missingFile("config.json") }
        let config = try JSONDecoder().decode(Qwen3ForcedAlignerConfig.self, from: Data(contentsOf: configURL))
        let aligner = Qwen3ForcedAligner(config)
        let weights = Qwen3ASRWeights.sanitize(try Qwen3ASRWeights.load(directory: directory), tied: false)
        if let q = config.base.perLayerQuantization {
            quantize(model: aligner) { path, _ in
                guard !path.hasPrefix("audio_tower"), weights["\(path).scales"] != nil else { return nil }
                return q.quantization(layer: path)?.asTuple
            }
        }
        let expected = Set(aligner.parameters().flattened().map { $0.0 })
        let provided = Set(weights.keys)
        let missing = expected.subtracting(provided).sorted(), unused = provided.subtracting(expected).sorted()
        guard missing.isEmpty, unused.isEmpty else { throw Qwen3ASRError.keyContract(missing: missing, unused: unused) }
        var params = weights
        if let dtype {
            for (k, v) in params where v.dtype != .uint32 && !k.hasSuffix(".scales") && !k.hasSuffix(".biases") { params[k] = v.asType(dtype) }
        }
        try aligner.update(parameters: ModuleParameters.unflattened(params), verify: .all)
        eval(aligner)
        aligner.wordTokenizer = try Qwen2WordTokenizer(directory: directory)
        return aligner
    }

    // MARK: - Words and prompt (ForceAlignProcessor)

    static func isKept(_ ch: Character) -> Bool {
        ch == "'" || ch.unicodeScalars.allSatisfy { s in
            let p = s.properties.generalCategory
            switch p {
            case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
                 .decimalNumber, .letterNumber, .otherNumber: return true
            default: return false
            }
        }
    }

    static func isCJK(_ ch: Character) -> Bool {
        guard let v = ch.unicodeScalars.first?.value else { return false }
        return (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v) || (0x20000...0x2A6DF).contains(v)
            || (0x2A700...0x2B73F).contains(v) || (0x2B740...0x2B81F).contains(v) || (0x2B820...0x2CEAF).contains(v)
            || (0xF900...0xFAFF).contains(v)
    }

    /// The units the aligner times (`tokenize_space_lang` / `tokenize_chinese_mixed`): whitespace words cleaned to
    /// letters, digits and apostrophes, each CJK ideograph its own unit.
    public static func wordList(_ text: String, language: String = "English") -> [String] {
        var out: [String] = []
        if language.lowercased() == "chinese" {
            var latin = ""
            func flush() { let c = String(latin.filter(isKept)); if !c.isEmpty { out.append(c) }; latin = "" }
            for ch in text {
                if isCJK(ch) { flush(); out.append(String(ch)) } else if isKept(ch) { latin.append(ch) } else { flush() }
            }
            flush()
            return out
        }
        for seg in text.split(whereSeparator: { $0.isWhitespace }) {
            let cleaned = String(seg.filter(isKept))
            guard !cleaned.isEmpty else { continue }
            var buf = ""
            for ch in cleaned {
                if isCJK(ch) { if !buf.isEmpty { out.append(buf); buf = "" }; out.append(String(ch)) } else { buf.append(ch) }
            }
            if !buf.isEmpty { out.append(buf) }
        }
        return out
    }

    /// `<|audio_start|>` + pads + `<|audio_end|>` + word₁ `<timestamp><timestamp>` word₂ … `<timestamp><timestamp>`,
    /// tokenized as the HF tokenizer does it: special tokens split the text, each word segment is BPE'd on its own.
    public func promptIDs(words: [String], audioTokens: Int) -> [Int] {
        guard let tok = wordTokenizer else { fatalError("word tokenizer not loaded") }
        let c = config.base, ts = config.timestampTokenId
        var ids = [c.audioStartTokenId] + Array(repeating: c.audioTokenId, count: audioTokens) + [c.audioEndTokenId]
        for w in words { ids += tok.encode(w); ids += [ts, ts] }
        return ids
    }

    /// `fix_timestamp`: keep a longest non-decreasing subsequence; repair runs of ≤ 2 outliers with the nearer valid
    /// neighbour, longer runs by linear interpolation.
    public static func fixTimestamps(_ data: [Double]) -> [Int] {
        let n = data.count
        guard n > 0 else { return [] }
        var tails: [Double] = [], levels: [[Int]] = [[]]
        for (i, v) in data.enumerated() {
            if v.isNaN { levels[0].append(i); continue }
            var lo = 0, hi = tails.count  // bisect_right
            while lo < hi { let mid = (lo + hi) / 2; if tails[mid] <= v { lo = mid + 1 } else { hi = mid } }
            if lo == tails.count { tails.append(v); if lo == levels.count { levels.append([]) } } else { tails[lo] = v }
            levels[lo].append(i)
        }
        var normal = [Bool](repeating: false, count: n)
        var idx = levels[levels.count - 1][0]
        normal[idx] = true
        for level in levels.dropLast().reversed() {
            idx = level.first(where: { data[$0] <= data[idx] })!
            normal[idx] = true
        }
        var result = data
        var i = 0
        while i < n {
            guard !normal[i] else { i += 1; continue }
            var j = i
            while j < n && !normal[j] { j += 1 }
            let count = j - i
            let left = (0..<i).reversed().first(where: { normal[$0] }).map { result[$0] }
            let right = (j..<n).first(where: { normal[$0] }).map { result[$0] }
            if count <= 2 {
                for k in i..<j {
                    if let l = left, let r = right { result[k] = (k - (i - 1)) <= (j - k) ? l : r } else { result[k] = left ?? right ?? result[k] }
                }
            } else if let l = left, let r = right {
                let step = (r - l) / Double(count + 1)
                for k in i..<j { result[k] = l + step * Double(k - i + 1) }
            } else if let v = left ?? right {
                for k in i..<j { result[k] = v }
            }
            i = j
        }
        return result.map { Int($0) }
    }

    // MARK: - Align

    /// Timestamp classes at the `<timestamp>` rows (argmax), for 16 kHz mono `samples` and the given words.
    public func timestampClasses(samples16k: [Float], words: [String]) -> (ids: [Int], classes: [Int]) {
        let feats = audioTower.encode(MLXArray(mel.features(samples16k).data).reshaped(-1, mel.nMels))
        let ids = promptIDs(words: words, audioTokens: feats.dim(0))
        guard let start = ids.firstIndex(of: config.base.audioTokenId) else { return (ids, []) }
        let n = feats.dim(0)
        let emb = model.embed(MLXArray(ids.map(Int32.init)))
        let input = concatenated([emb[0 ..< start], feats.asType(emb.dtype), emb[(start + n)...]], axis: 0).expandedDimensions(axis: 0)
        let logits = lmHead(model(embeddings: input, cache: nil))  // [1, L, classify_num]
        let mask = ids.enumerated().filter { $0.element == config.timestampTokenId }.map { Int32($0.offset) }
        let picked = argMax(logits[0].take(MLXArray(mask), axis: 0), axis: -1)
        return (ids, picked.asArray(Int32.self).map(Int.init))
    }

    /// Word timings of `text` in mono `samples` at any `sampleRate` (resampled to 16 kHz by AVAudioConverter). Times
    /// are seconds from the first sample. For a TTS stream, call it once per sentence chunk with that chunk's text and
    /// add the chunk's start time.
    public func align(samples: [Float], sampleRate: Int, text: String, language: String = "English") throws -> [AlignedWord] {
        try align(samples16k: Self.resample(samples, from: sampleRate), text: text, language: language)
    }

    /// Mono float resampling to 16 kHz (AVAudioConverter, flushed to the end of the input).
    public static func resample(_ samples: [Float], from rate: Int) throws -> [Float] {
        guard rate != 16000, !samples.isEmpty else { return samples }
        guard let inF = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(rate), channels: 1, interleaved: false),
              let outF = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: inF, to: outF),
              let inBuf = AVAudioPCMBuffer(pcmFormat: inF, frameCapacity: AVAudioFrameCount(samples.count)),
              let outBuf = AVAudioPCMBuffer(pcmFormat: outF, frameCapacity: AVAudioFrameCount(Double(samples.count) * 16000 / Double(rate)) + 1024)
        else { throw Qwen3ASRError.missingFile("an audio converter for \(rate) Hz") }
        inBuf.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { inBuf.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        var supplied = false
        var error: NSError?
        _ = conv.convert(to: outBuf, error: &error) { _, status in
            if supplied { status.pointee = .endOfStream; return nil }
            supplied = true; status.pointee = .haveData; return inBuf
        }
        if let error { throw error }
        return Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: Int(outBuf.frameLength)))
    }

    /// Word timings of `text` in 16 kHz mono `samples` (`generate` for one clip).
    public func align(samples16k: [Float], text: String, language: String = "English") -> [AlignedWord] {
        let words = Self.wordList(text, language: language)
        guard !words.isEmpty, !samples16k.isEmpty else { return [] }
        let (_, classes) = timestampClasses(samples16k: samples16k, words: words)
        let fixed = Self.fixTimestamps(classes.map { Double($0) * config.timestampSegmentTime })
        return words.enumerated().map { k, w in
            AlignedWord(text: w, start: Double(fixed[2 * k]) / 1000, end: Double(fixed[2 * k + 1]) / 1000)
        }
    }
}

/// Qwen2's byte-level BPE for plain word segments, from `vocab.json` + `merges.txt` (the forced-aligner repos ship no
/// `tokenizer.json`): NFC, the Qwen2 pre-tokenizer regex, GPT-2 byte mapping, merges by rank. Special tokens are not
/// handled here; callers insert their ids.
public final class Qwen2WordTokenizer {
    static let pattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    let vocab: [String: Int]
    let ranks: [String: Int]
    let split: NSRegularExpression
    static let byteChars: [Character] = {
        let keep = Array(33...126) + Array(161...172) + Array(174...255)
        var map = [Int: Int]()
        for b in keep { map[b] = b }
        var n = 0
        for b in 0..<256 where map[b] == nil { map[b] = 256 + n; n += 1 }
        return (0..<256).map { Character(UnicodeScalar(UInt32(map[$0]!))!) }
    }()
    private var cache: [String: [Int]] = [:]

    public init(directory: URL) throws {
        let vURL = directory.appendingPathComponent("vocab.json"), mURL = directory.appendingPathComponent("merges.txt")
        guard let v = try JSONSerialization.jsonObject(with: Data(contentsOf: vURL)) as? [String: Any] else { throw Qwen3ASRError.missingFile("vocab.json") }
        vocab = v.compactMapValues { ($0 as? NSNumber)?.intValue }
        var r = [String: Int]()
        var rank = 0
        for line in try String(contentsOf: mURL, encoding: .utf8).split(separator: "\n", omittingEmptySubsequences: true) {
            if line.hasPrefix("#version") { continue }
            r[String(line)] = rank; rank += 1
        }
        ranks = r
        split = try NSRegularExpression(pattern: Self.pattern)
    }

    private func bpe(_ piece: String) -> [Int] {
        if let hit = cache[piece] { return hit }
        var word = piece.utf8.map { String(Self.byteChars[Int($0)]) }
        while word.count > 1 {
            var best = Int.max, at = -1
            for i in 0..<(word.count - 1) { if let rk = ranks[word[i] + " " + word[i + 1]], rk < best { best = rk; at = i } }
            if at < 0 { break }
            let a = word[at], b = word[at + 1]
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == a && word[i + 1] == b { merged.append(a + b); i += 2 } else { merged.append(word[i]); i += 1 }
            }
            word = merged
        }
        let ids = word.compactMap { vocab[$0] }
        cache[piece] = ids
        return ids
    }

    public func encode(_ text: String) -> [Int] {
        let t = text.precomposedStringWithCanonicalMapping as NSString
        var ids: [Int] = []
        var pos = 0
        for m in split.matches(in: t as String, range: NSRange(location: 0, length: t.length)) where m.range.length > 0 {
            if m.range.location > pos { ids += bpe(t.substring(with: NSRange(location: pos, length: m.range.location - pos))) }
            ids += bpe(t.substring(with: m.range))
            pos = m.range.location + m.range.length
        }
        if pos < t.length { ids += bpe(t.substring(from: pos)) }
        return ids
    }
}
