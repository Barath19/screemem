# macOS menu-bar bridge (Swift)

Lets a native macOS menu-bar vision app contribute to screemem's memory.

The app in question runs a local Gemma vision model (`llama-server` +
`mmproj` on 127.0.0.1:8080) and lets you shake your mouse to ask about whatever is
on screen. On its own it answers and forgets. This bridge keeps the answer.

After applying: **shake your mouse → ask about your screen → that observation
becomes part of the same cognee graph as your team's Slack threads and GitHub
issues**, answerable weeks later and from Slack.

It is one new file plus a two-hunk patch, deliberately — the host app stays its own
project and this stays trivial to rebase when it moves.

## Apply

```bash
cd <the-menubar-app>
cp /path/to/screemem/genielm/MemoryBridge.swift Sources/GenieLM/
git apply /path/to/screemem/genielm/main.swift.patch
swift build -c release

# swap the new binary into the existing bundle and re-sign
cp .build/release/GenieLM GenieLM.app/Contents/MacOS/GenieLM
codesign --force --deep -s - GenieLM.app
open GenieLM.app
```

screemem's `app.py` must be running — it exposes the receiving endpoint
(`POST /api/v1/memory/screen`) and owns the graph lock.

## What the patch does

Two hunks in `Sources/GenieLM/main.swift`:

1. Captures `turnHadImage` **before** `history` is mutated, because the check for
   whether a screenshot was attached (`history.isEmpty && pendingImageB64 != nil`)
   stops being true once the first message is appended.
2. Calls `MemoryBridge.record(...)` after a successful answer, next to the
   existing `RetroSound.answer()`.

## Design notes

**The screenshot never crosses the wire.** Only the question text and the model's
answer are sent. The image stays on the machine — the same rule the Python capture
path follows, and the reason there is no hosted-vision fallback anywhere in this
project.

**Only turns that carried a screenshot are recorded.** A text-only follow-up
observes nothing new about the screen; storing it would fill the graph with chat
chatter that competes with real observations during retrieval.

**Fire-and-forget.** The POST is fully asynchronous and every failure path is
swallowed to a log line. If screemem is not running, the host app must behave
exactly as it did before — a memory layer that can break the app it is attached to
is not worth having.

**Answers shorter than 20 characters are dropped** ("yes", "a terminal") — they
carry no findable detail.

## Configuration

| Variable | Effect |
|---|---|
| `SCREEMEM_URL` | screemem base URL (default `http://127.0.0.1:8000`) |
| `SCREEMEM_DISABLED=1` | stop contributing memory without rebuilding |

## Security

The receiving endpoint is **unauthenticated and local-only by design** — it is
reached over loopback from an app running as the same user, so a signature scheme
would add ceremony without adding a boundary. Do not expose port 8000. The Slack
endpoint in the same server *is* signature-verified, because that one takes
requests from the internet.
