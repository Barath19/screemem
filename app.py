"""Slack front end for the Meridian memory.

    /meridian-ask <question>              ask across every source
    /meridian-ask slack: <question>       scope to Slack only
    /meridian-ask github: <question>      scope to GitHub only
    /meridian-remember <fact>             add something Slack never saw

Commands are namespaced with `meridian-` on purpose. At a hackathon where every
team installs an app from the same manifest, several apps register the identical
/cognee-ask command; Slack then shows a disambiguation dropdown and your command
can silently reach another team's laptop. That failure looks exactly like your
own backend being broken.
"""

import hashlib
import hmac
import os
import ssl
import time

import aiohttp
import certifi
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from memory import DATASET, SOURCE_SLACK, setup
from query import ask

setup()

SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
# macOS framework Python ships without system CA certs wired up, and aiohttp
# (unlike httpx) does not bundle certifi. Without this, posting the real answer
# back to hooks.slack.com dies with SSLCertVerificationError.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

app = FastAPI(title="Meridian memory bot")


def verify_slack_signature(body: bytes, timestamp: str, signature: str, secret: str) -> bool:
    if not timestamp or not signature or not secret:
        return False
    try:
        if abs(time.time() - float(timestamp)) > 60 * 5:
            return False
    except ValueError:
        return False
    basestring = f"v0:{timestamp}:".encode() + body
    digest = "v0=" + hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)


def parse_scope(text: str) -> tuple[str, str | None]:
    """`slack: why did we...` -> ("why did we...", "slack")"""
    for source in ("slack", "github"):
        prefix = f"{source}:"
        if text.lower().startswith(prefix):
            return text[len(prefix) :].strip(), source
    return text, None


def parse_deep(text: str) -> tuple[str, bool]:
    """`deep: why did we...` -> ("why did we...", True)"""
    if text.lower().startswith("deep:"):
        return text[len("deep:") :].strip(), True
    return text, False


def format_agent_answer(a) -> dict:
    """Block Kit for the multi-agent pipeline. The plan and the verifier tally are
    shown deliberately — the point of this path is that its work is inspectable."""
    plan_lines = "\n".join(
        f"{i}. `[{s.source}]` {s.sub_question}" for i, s in enumerate(a.plan, 1)
    )
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{a.question}*"}},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Retrieval plan*\n{plan_lines}"},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": a.text[:2900]}},
    ]

    if a.dropped:
        dropped = "\n".join(f"• {d}" for d in a.dropped[:4])
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Dropped — unsupported by evidence:*\n{dropped}",
                },
            }
        )

    if a.references:
        cited = "\n".join(f"• {r}" for r in a.references)
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Grounded in:*\n{cited}"}}
        )

    if not a.grounded:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":warning: _Not a recollection. Either no claim "
                    "survived verification, or the memory has never heard of the "
                    "subject at all._",
                },
            }
        )

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"multi-agent · {len(a.plan)} retriever(s) · "
                    f"{a.supported} claim(s) verified, {a.unsupported} rejected · "
                    f"{a.seconds:.1f}s",
                }
            ],
        }
    )
    return {"response_type": "ephemeral", "blocks": blocks, "text": a.text[:200]}


def format_answer(a) -> dict:
    """Slack Block Kit. The provenance footer is the point: an answer you can
    trace back to a thread or an issue is a different object from a chat reply."""
    header = f"*{a.question}*"
    if a.scope:
        header += f"   _(scoped to {a.scope} only)_"

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": header}},
        {"type": "section", "text": {"type": "mrkdwn", "text": a.text[:2900]}},
    ]

    if a.references:
        cited = "\n".join(f"• {r}" for r in a.references)
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Grounded in:*\n{cited}"}}
        )

    if not a.grounded:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":warning: _No supporting evidence in the graph — "
                    "treat this as a guess, not a recollection._",
                },
            }
        )

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"routed to `{a.search_type}` · {a.seconds:.1f}s · "
                    f"{len(a.references)} graph reference(s)",
                }
            ],
        }
    )
    return {"response_type": "ephemeral", "blocks": blocks, "text": a.text[:200]}


async def handle_command(command: str, text: str, user_name: str, channel: str) -> dict:
    text = text.strip()
    if not text:
        return {"response_type": "ephemeral", "text": f"Usage: `{command} <text>`"}

    if command.endswith("-ask"):
        question, deep = parse_deep(text)
        question, source = parse_scope(question)

        if deep:
            from agents import ask_with_agents

            return format_agent_answer(await ask_with_agents(question))

        answer = await ask(question, source=source)
        return format_answer(answer)

    if command.endswith("-remember"):
        import cognee

        # Provenance goes into the document text, not just metadata — the graph
        # extraction is an LLM reading the body, so anything left in metadata is
        # invisible to it. Same shape ingest.py writes for exported threads.
        stamped = (
            f"Fact recorded in Slack on {time.strftime('%A %d %B %Y')} "
            f"by {user_name} in #{channel}.\n\n{text}"
        )
        await cognee.add(stamped, dataset_name=DATASET, node_set=[SOURCE_SLACK, "manual"])
        await cognee.cognify(datasets=[DATASET])
        return {
            "response_type": "ephemeral",
            "text": f":brain: Remembered, and folded into the graph:\n> {text}",
        }

    return {"response_type": "ephemeral", "text": f"Unknown command: {command}"}


async def run_and_post(command: str, text: str, user_name: str, channel: str, url: str) -> None:
    try:
        reply = await handle_command(command, text, user_name, channel)
    except Exception as exc:  # never leave the user staring at "Working on it..."
        reply = {"response_type": "ephemeral", "text": f"Something went wrong: {exc}"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=reply, ssl=SSL_CONTEXT)


@app.post("/api/v1/slack/commands")
async def slack_commands(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    if not verify_slack_signature(
        body,
        request.headers.get("X-Slack-Request-Timestamp", ""),
        request.headers.get("X-Slack-Signature", ""),
        SIGNING_SECRET,
    ):
        raise HTTPException(status_code=401, detail="invalid signature")

    form = await request.form()
    command = form.get("command", "")
    text = form.get("text", "")
    response_url = form.get("response_url", "")
    user_name = form.get("user_name", "someone")
    channel = form.get("channel_name", "unknown")

    # Slack hangs up at 3s. Graph completion takes ~4s and temporal ~40s, so the
    # real work always goes to the background and comes back via response_url.
    if response_url:
        background_tasks.add_task(
            run_and_post, command, text, user_name, channel, response_url
        )
        verb = "Recalling" if command.endswith("-ask") else "Remembering"
        return JSONResponse({"response_type": "ephemeral", "text": f":brain: {verb}…"})

    return JSONResponse(await handle_command(command, text, user_name, channel))


@app.post("/api/v1/memory/screen")
async def remember_screen(request: Request):
    """Ingest a screen observation.

    This endpoint exists because of the graph lock. A running cognee process
    holds an exclusive lock on the graph store, so a separate capture loop cannot
    write while this server is up. Routing writes through the process that
    already owns the lock lets continuous capture and Slack answering coexist —
    and gives GenieLM (or anything else) a one-line way to contribute memory.

    Local-only by design: no signature check, so do not expose this port.
    """
    body = await request.json()
    summary = (body.get("summary") or "").strip()
    if not summary:
        return JSONResponse({"ok": False, "error": "summary is required"}, status_code=400)

    app_name = (body.get("app") or "unknown").strip()
    when = body.get("when") or time.strftime("%A %d %B %Y at %H:%M")

    # Same provenance-in-the-text rule as every other ingester here.
    document = (
        f"Screen observation on {when}.\n"
        f"Source: screen capture on this machine, frontmost application {app_name}.\n\n"
        f"{summary}"
    )

    import cognee

    await cognee.add(document, dataset_name=DATASET, node_set=["screen", "manual"])
    await cognee.cognify(datasets=[DATASET], chunk_size=2048)

    result = {"ok": True, "stored": len(document), "app": app_name, "when": when}

    # "add to cognee cloud" mirrors the freshly-built graph up to the tenant so it
    # is visible in the Cloud UI. The local graph stays the source of truth: push()
    # ships nodes and edges but not the vector index, so the cloud copy is for
    # looking at, not for querying — a cloud-side question retrieves nothing and
    # the remote LLM answers from thin air.
    if body.get("push"):
        if not (os.environ.get("COGNEE_SERVICE_URL") and os.environ.get("COGNEE_API_KEY")):
            result["push"] = "skipped: COGNEE_SERVICE_URL / COGNEE_API_KEY not set"
        else:
            try:
                pushed = await cognee.push(dataset=DATASET, target_dataset=DATASET)
                fields = getattr(pushed, "__dict__", {}) or {}
                result["push"] = {
                    k: v for k, v in fields.items()
                    if k in ("status", "num_nodes", "num_edges", "dataset_name")
                }
            except Exception as exc:  # never fail the local write because the cloud failed
                result["push"] = f"failed: {exc}"

    return result


@app.get("/memory")
async def browse_memory():
    """Every document in the memory, newest source first, as a plain page.

    Exists because the stored text is otherwise only visible by querying Qdrant
    by hand. Reads the vector store directly rather than the graph so it works
    while the graph lock is held by this same process.
    """
    import html as _html
    import json as _json
    import urllib.request as _u

    req = _u.Request(
        "http://localhost:6333/collections/DocumentChunk_text/points/scroll",
        data=_json.dumps({"limit": 500, "with_payload": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        points = _json.load(_u.urlopen(req, timeout=20))["result"]["points"]
    except Exception as exc:
        return HTMLResponse(f"<pre>could not read Qdrant: {_html.escape(str(exc))}</pre>", 500)

    buckets: dict[str, list[str]] = {
        "Screen observations": [],
        "Slack threads": [],
        "GitHub issues": [],
        "Added manually": [],
    }
    for p in points:
        text = (p.get("payload", {}).get("text") or "").strip()
        if not text:
            continue
        if text.startswith(("Screen observation", "Added to cognee")):
            buckets["Screen observations"].append(text)
        elif text.startswith("Slack thread"):
            buckets["Slack threads"].append(text)
        elif text.startswith("GitHub issue"):
            buckets["GitHub issues"].append(text)
        else:
            buckets["Added manually"].append(text)

    css = """
    :root { color-scheme: light dark }
    body { font: 15px/1.55 ui-sans-serif,-apple-system,Segoe UI,sans-serif;
           max-width: 60rem; margin: 2rem auto; padding: 0 1.25rem }
    h1 { font-size: 1.5rem; margin-bottom: .25rem }
    .sub { opacity:.65; margin-bottom: 2rem }
    h2 { font-size: 1.05rem; margin: 2rem 0 .75rem; padding-bottom:.35rem;
         border-bottom: 1px solid color-mix(in oklab, currentColor 20%, transparent) }
    .count { opacity:.55; font-weight: 400 }
    article { border: 1px solid color-mix(in oklab, currentColor 15%, transparent);
              border-radius: 10px; padding: .85rem 1rem; margin-bottom: .7rem }
    article.screen { border-color: color-mix(in oklab, #16a34a 55%, transparent) }
    pre { white-space: pre-wrap; word-wrap: break-word; margin: 0; font: inherit }
    """

    parts = [
        f"<style>{css}</style>",
        "<h1>screemem — everything in the memory</h1>",
        f"<div class=sub>{len(points)} documents · one cognee graph · "
        "vector search in Qdrant · text only, no images are ever stored</div>",
    ]
    for title, docs in buckets.items():
        if not docs:
            continue
        cls = " class=screen" if title == "Screen observations" else ""
        parts.append(f"<h2>{title} <span class=count>· {len(docs)}</span></h2>")
        for d in sorted(docs, reverse=True):
            parts.append(f"<article{cls}><pre>{_html.escape(d)}</pre></article>")

    return HTMLResponse("\n".join(parts))


@app.get("/health")
async def health():
    return {"ok": True, "dataset": DATASET, "signing_secret_set": bool(SIGNING_SECRET)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
