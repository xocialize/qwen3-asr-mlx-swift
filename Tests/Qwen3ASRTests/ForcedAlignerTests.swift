import XCTest
@testable import Qwen3ASR

/// The forced aligner's pure parts against mlx-audio's ForceAlignProcessor (values worked by hand from the Python).
final class ForcedAlignerTests: XCTestCase {
    func testWordListSpaceLanguage() {
        XCTAssertEqual(Qwen3ForcedAligner.wordList("We don't have much time. The bridge — at 12:30!"),
                       ["We", "don't", "have", "much", "time", "The", "bridge", "at", "1230"])
        XCTAssertEqual(Qwen3ForcedAligner.wordList("Hello 世界 ok"), ["Hello", "世", "界", "ok"])
    }

    func testWordListChinese() {
        XCTAssertEqual(Qwen3ForcedAligner.wordList("我们没有ABC时间。", language: "Chinese"), ["我", "们", "没", "有", "ABC", "时", "间"])
    }

    func testFixTimestamps() {
        // monotone input is untouched
        XCTAssertEqual(Qwen3ForcedAligner.fixTimestamps([0, 80, 80, 160]), [0, 80, 80, 160])
        // one outlier: replaced by the nearer valid neighbour (left on a tie)
        XCTAssertEqual(Qwen3ForcedAligner.fixTimestamps([0, 80, 900, 160, 240]), [0, 80, 80, 160, 240])
        // the longest non-decreasing run wins even when it is the wrong one; the trailing drop takes its neighbour
        XCTAssertEqual(Qwen3ForcedAligner.fixTimestamps([0, 900, 950, 990, 400]), [0, 900, 950, 990, 990])
        // a run of three outliers: interpolated between the neighbours
        XCTAssertEqual(Qwen3ForcedAligner.fixTimestamps([0, 900, 800, 700, 100, 200]), [0, 25, 50, 75, 100, 200])
    }
}
