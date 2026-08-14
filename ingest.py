"""Turn a Slack export and a GitHub issue dump into one cognee knowledge graph.

    python ingest.py                    # ingest corpus/ then build the graph
    python ingest.py --slack-only       # skip GitHub (to demo the "before" state)
    python ingest.py --dry-run          # print the documents, call nothing

Two design decisions worth knowing, because both are load-bearing:

1. ONE THREAD IS ONE DOCUMENT, not one message. A single Slack message is
   usually meaningless on its own ("Paused or dead?"); the thread is the unit
   that carries an actual claim. Chunking per message would also multiply the
   cognify LLM calls by ~5x for no gain.

2. THE DATE AND AUTHOR GO INTO THE DOCUMENT TEXT, not just into metadata.
   cognee's graph extraction is an LLM reading the document body. Slack's `ts`
   epoch field and the channel name live in the JSON envelope, which the LLM
   never sees. If you ingest bare message text you get a graph with no idea who
   said anything or when — and every time-scoped question silently degrades.
   So we render provenance into the prose.
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from memory import DATASET, SOURCE_GITHUB, SOURCE_SLACK, require_llm_key, setup

CORPUS = Path(__file__).parent / "corpus"
SLACK_EXPORT = CORPUS / "slack_export"
GITHUB_ISSUES = CORPUS / "github_issues.json"
REPO = "meridian/platform"


# --------------------------------------------------------------------------
# Slack export parsing
# --------------------------------------------------------------------------
# Real Slack exports are a directory per channel containing one JSON file per
# day, plus users.json / channels.json at the root. This parser targets that
# layout, so pointing it at an actual workspace export works unchanged.


def load_users(export_dir: Path) -> dict[str, str]:
    users_file = export_dir / "users.json"
    if not users_file.exists():
        return {}
    users = json.loads(users_file.read_text())
    return {
        u["id"]: (u.get("profile", {}).get("real_name") or u.get("name") or u["id"])
        for u in users
    }


def fmt_ts(ts: str) -> str:
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M")


def fmt_date(ts: str) -> str:
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.strftime("%A %d %B %Y")


def slack_threads(export_dir: Path) -> list[dict]:
    """Group an export into threads. A top-level message with no replies is
    still a thread of one — standalone messages often carry the decision."""
    users = load_users(export_dir)
    threads: list[dict] = []

    for channel_dir in sorted(p for p in export_dir.iterdir() if p.is_dir()):
        channel = channel_dir.name
        messages: list[dict] = []
        for day_file in sorted(channel_dir.glob("*.json")):
            messages.extend(json.loads(day_file.read_text()))

        # Bucket by thread parent. Slack marks replies with thread_ts pointing
        # at the parent's ts; parents either omit it or repeat their own ts.
        buckets: dict[str, list[dict]] = {}
        for m in messages:
            if m.get("type") != "message" or not m.get("text"):
                continue
            key = m.get("thread_ts") or m["ts"]
            buckets.setdefault(key, []).append(m)

        for parent_ts, msgs in buckets.items():
            msgs.sort(key=lambda m: float(m["ts"]))
            threads.append(
                {
                    "channel": channel,
                    "ts": parent_ts,
                    "messages": msgs,
                    "participants": [users.get(m["user"], m["user"]) for m in msgs],
                }
            )

    threads.sort(key=lambda t: float(t["ts"]))
    return threads


def render_thread(thread: dict, users: dict[str, str]) -> str:
    channel = thread["channel"]
    opened = thread["messages"][0]["ts"]
    # dict.fromkeys to dedupe while keeping speaking order
    who = list(dict.fromkeys(thread["participants"]))

    lines = [
        f"Slack thread in #{channel}, {fmt_date(opened)}.",
        f"Source: Slack workspace export, channel #{channel}.",
        f"Participants: {', '.join(who)}.",
        "",
    ]
    for i, m in enumerate(thread["messages"]):
        speaker = users.get(m["user"], m["user"])
        marker = "" if i == 0 else " (in thread)"
        lines.append(f"[{fmt_ts(m['ts'])}] {speaker}{marker}: {m['text']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# GitHub issue rendering
# --------------------------------------------------------------------------


def render_issue(issue: dict) -> str:
    assignees = ", ".join(issue.get("assignees") or []) or "nobody"
    labels = ", ".join(issue.get("labels") or []) or "none"
    lines = [
        f'GitHub issue {REPO}#{issue["number"]}: {issue["title"]}',
        f"Source: GitHub issue tracker for the repository {REPO}.",
        f'State: {issue["state"]}. Opened {issue["created_at"]} by {issue["author"]}.',
        f"Assigned to: {assignees}. Labels: {labels}.",
    ]
    if issue.get("closed_at"):
        lines.append(f'Closed {issue["closed_at"]}.')
    lines += ["", issue["body"]]
    for c in issue.get("comments") or []:
        lines += ["", f'Comment by {c["author"]} on {c["created_at"]}: {c["body"]}']
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


def build_documents(slack_only: bool = False) -> list[tuple[str, list[str], str]]:
    """Returns (text, node_set, label) triples."""
    docs: list[tuple[str, list[str], str]] = []

    users = load_users(SLACK_EXPORT)
    for t in slack_threads(SLACK_EXPORT):
        docs.append(
            (
                render_thread(t, users),
                [SOURCE_SLACK, f"channel-{t['channel']}"],
                f"slack #{t['channel']} {fmt_ts(t['ts'])}",
            )
        )

    if not slack_only:
        for issue in json.loads(GITHUB_ISSUES.read_text()):
            docs.append(
                (
                    render_issue(issue),
                    [SOURCE_GITHUB, f"issue-{issue['number']}"],
                    f"github #{issue['number']} {issue['title'][:40]}",
                )
            )

    return docs


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slack-only", action="store_true", help="skip GitHub issues")
    ap.add_argument("--dry-run", action="store_true", help="print documents, call nothing")
    ap.add_argument("--reset", action="store_true", help="wipe memory before ingesting")
    args = ap.parse_args()

    docs = build_documents(slack_only=args.slack_only)

    if args.dry_run:
        for text, node_set, label in docs:
            print(f"\n{'=' * 78}\n{label}   node_set={node_set}\n{'=' * 78}\n{text}")
        print(f"\n{len(docs)} documents, {sum(len(d[0]) for d in docs):,} chars")
        return

    setup()
    require_llm_key()
    import cognee

    if args.reset:
        print("wiping existing memory...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

    print(f"adding {len(docs)} documents to dataset '{DATASET}'...")
    for text, node_set, label in docs:
        await cognee.add(text, dataset_name=DATASET, node_set=node_set)
        print(f"  + {label}")

    # One cognify run over the whole dataset rather than per-document. This is
    # the expensive step: it is an LLM pass per chunk to extract entities and
    # relationships, so it dominates wall-clock and cost.
    print("\nbuilding the knowledge graph (this is the slow part)...")
    # chunk_size is set high enough that every document is a single chunk. Each
    # chunk costs an entity-extraction LLM call plus a summarisation call, so
    # chunking a 900-character Slack thread into three pieces triples the bill
    # and fragments the entities across chunks for no benefit. Our longest
    # document (GitHub #412) is well under this.
    await cognee.cognify(datasets=[DATASET], chunk_size=2048)
    print("done. try:  python ask.py \"why did we pause the Neon migration?\"")


if __name__ == "__main__":
    asyncio.run(main())
