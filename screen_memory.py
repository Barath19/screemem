"""Screen -> on-device vision summary -> cognee graph -> answerable from Slack.

    python screen_memory.py                 # capture once, summarise, remember
    python screen_memory.py --watch 300     # every 5 minutes
    python screen_memory.py --no-store      # summarise and print, store nothing

Screen-grounded chat, pointed at memory instead of at the moment: a local vision
model looks at your screen and the observation is kept, so the answer is still
available next week — and to your team, from Slack.

Privacy is the reason it is built this way round. Summarisation runs against a
local `llama-server` (gemma-3-4b + mmproj) on 127.0.0.1:8080, so
**the screenshot never leaves the machine**. Only the short text summary is
stored, and the PNG is deleted immediately unless --keep is passed. Sending raw
screenshots to a hosted vision API would be one line shorter and considerably
worse; there is no cloud fallback for the image on purpose.
"""

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from memory import DATASET, setup

LLAMA = "http://127.0.0.1:8080"
SOURCE_SCREEN = "screen"

PROMPT = """Look at this screenshot of someone's computer screen and write a short
factual note for a work log. Cover, in at most 4 sentences:
- which application or website is in focus
- what specific task the person appears to be doing
- any concrete identifiers visible that would make this findable later: file
  names, function or class names, error messages, ticket or PR numbers, branch
  names, people's names, URLs

Write plain prose. Do not speculate about intent beyond what is visible, do not
describe window decorations or wallpaper, and do not editorialise. If the screen
is idle or shows nothing meaningful, reply exactly: NOTHING OF NOTE."""


def frontmost_app() -> str:
    """Ask the window server what is in focus — cheaper and more reliable than
    asking a vision model to read the menu bar."""
    try:
        out = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first process '
             "whose frontmost is true"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def capture(path: Path) -> bool:
    """-x suppresses the shutter sound so a watch loop is not audible."""
    r = subprocess.run(["screencapture", "-x", "-t", "png", str(path)], capture_output=True)
    return r.returncode == 0 and path.exists() and path.stat().st_size > 0


def summarise(png: Path) -> str:
    b64 = base64.b64encode(png.read_bytes()).decode()
    body = json.dumps({
        "model": "local",
        "temperature": 0,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    }).encode()

    req = urllib.request.Request(
        f"{LLAMA}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            payload = json.load(r)
    except urllib.error.URLError as e:
        sys.exit(
            f"Could not reach the local vision model at {LLAMA} ({e}).\n"
            "Start it with:\n"
            '  M="$HOME/Library/Application Support/GenieLM/models"\n'
            '  llama-server -m "$M/gemma-3-4b-it-Q4_K_M.gguf" '
            '--mmproj "$M/mmproj-model-f16.gguf" --port 8080 -ngl 99 --jinja -c 8192'
        )
    return payload["choices"][0]["message"]["content"].strip()


def as_document(summary: str, app_name: str, when: datetime) -> str:
    """Same provenance-in-the-text rule the Slack and GitHub ingesters follow:
    graph extraction is an LLM reading the body, so the timestamp and the app
    name have to be prose, not metadata, or the graph cannot place this in time
    or attribute it to an application."""
    return (
        f"Screen observation on {when.strftime('%A %d %B %Y at %H:%M')}.\n"
        f"Source: screen capture on this machine, frontmost application {app_name}.\n\n"
        f"{summary}"
    )


SERVER = "http://127.0.0.1:8000"


def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{SERVER}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def remember_via_server(summary: str, app_name: str, when: str) -> bool:
    """Hand the write to app.py, which already holds the graph lock."""
    body = json.dumps({"summary": summary, "app": app_name, "when": when}).encode()
    req = urllib.request.Request(
        f"{SERVER}/api/v1/memory/screen", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r).get("ok", False)
    except Exception as e:
        print(f"  server write failed: {e}")
        return False


async def remember_direct(text: str) -> None:
    """Write straight to cognee. Only possible when no server is running — a
    live cognee process holds an exclusive lock on the graph store."""
    import cognee

    await cognee.add(text, dataset_name=DATASET, node_set=[SOURCE_SCREEN, "manual"])
    await cognee.cognify(datasets=[DATASET], chunk_size=2048)


async def once(store: bool, keep: bool) -> str | None:
    tmp = Path(tempfile.gettempdir()) / f"screen_{int(time.time())}.png"
    try:
        if not capture(tmp):
            print("capture failed — grant Screen Recording permission to your terminal")
            return None

        app_name = frontmost_app()
        summary = summarise(tmp)

        if summary.strip().upper().startswith("NOTHING OF NOTE"):
            print(f"[{datetime.now():%H:%M}] {app_name}: nothing of note, skipped")
            return None

        now = datetime.now()
        print(f"\n[{now:%H:%M}] {app_name}\n{summary}\n")

        if store:
            # Prefer the server: it owns the graph lock, so this is what makes
            # --watch work at the same time as the Slack bot.
            if server_up():
                ok = remember_via_server(summary, app_name,
                                         now.strftime("%A %d %B %Y at %H:%M"))
                print("stored via app.py — answerable from Slack" if ok
                      else "not stored")
            else:
                await remember_direct(as_document(summary, app_name, now))
                print("stored directly (no server running)")
        return summary
    finally:
        # The image is the sensitive artefact, not the summary. Delete it unless
        # explicitly asked to keep it.
        if not keep:
            tmp.unlink(missing_ok=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, metavar="SECONDS",
                    help="capture on a loop every N seconds")
    ap.add_argument("--no-store", action="store_true", help="summarise only, store nothing")
    ap.add_argument("--keep", action="store_true", help="keep the PNG on disk (debugging)")
    args = ap.parse_args()

    setup()
    store = not args.no_store

    if not args.watch:
        await once(store, args.keep)
        return

    print(f"watching every {args.watch}s — ctrl-c to stop")
    try:
        while True:
            await once(store, args.keep)
            await asyncio.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    asyncio.run(main())
