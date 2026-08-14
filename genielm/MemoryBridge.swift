import AppKit
import Foundation

/// Sends what GenieLM saw to screemem, so a shake-to-chat answer becomes part of
/// the team's memory instead of vanishing when the bubble closes.
///
/// GenieLM answers about your screen *now*. screemem keeps that answer, in the
/// same cognee knowledge graph as the team's Slack threads and GitHub issues, so
/// "what was I working on when I filed #1679" is answerable next week and from
/// Slack.
///
/// Only the text goes over the wire — never the screenshot. The image stays on
/// this machine, which is the same rule the rest of screemem follows.
///
/// Fire-and-forget by design: if screemem is not running, GenieLM must keep
/// working exactly as before. Nothing here is allowed to block the chat or
/// surface an error into the bubble.
enum MemoryBridge {

    /// Local screemem server. Override with SCREEMEM_URL.
    private static var endpoint: URL? {
        let base = ProcessInfo.processInfo.environment["SCREEMEM_URL"]
            ?? "http://127.0.0.1:8000"
        return URL(string: base + "/api/v1/memory/screen")
    }

    /// Set SCREEMEM_DISABLED=1 to stop contributing memory without rebuilding.
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["SCREEMEM_DISABLED"] != "1"
    }

    private static var frontmostApp: String {
        NSWorkspace.shared.frontmostApplication?.localizedName ?? "unknown"
    }

    private static var timestamp: String {
        let f = DateFormatter()
        f.dateFormat = "EEEE dd MMMM yyyy 'at' HH:mm"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f.string(from: Date())
    }

    /// Record one screen-grounded exchange.
    /// - Parameters:
    ///   - question: what the user asked.
    ///   - answer: what the local vision model replied.
    ///   - sawImage: true when a screenshot was actually attached. Text-only
    ///     follow-ups are skipped — they carry no new observation of the screen,
    ///     and storing them would fill the graph with chat chatter.
    static func record(question: String, answer: String, sawImage: Bool) {
        guard enabled, sawImage, let url = endpoint else { return }

        let a = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard a.count > 20 else { return }   // too short to be an observation

        // Phrased so the graph extractor reads it as an observation of the
        // screen, not as a conversation. The question is kept because it is
        // often the only thing naming what the user cared about.
        let summary = "Asked about the screen: \"\(q)\"\n\nWhat was on screen: \(a)"

        let payload: [String: Any] = [
            "summary": summary,
            "app": frontmostApp,
            "when": timestamp,
        ]
        guard let body = try? JSONSerialization.data(withJSONObject: payload) else { return }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        // cognify runs on the far side, so allow real time — but never block the UI.
        req.timeoutInterval = 300

        URLSession.shared.dataTask(with: req) { _, response, error in
            if let error {
                print("[screemem] not recorded: \(error.localizedDescription)")
            } else if let code = (response as? HTTPURLResponse)?.statusCode {
                print(code == 200 ? "[screemem] recorded" : "[screemem] server said \(code)")
            }
        }.resume()
    }
}
