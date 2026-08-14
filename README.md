# Meridian Memory — cross-source memory for Slack

A Slack bot that answers questions **whose answers are not in Slack**.

Slack threads and GitHub issues are ingested into a single cognee knowledge
graph, backed by Qdrant for vector search. Because both sources live in one
graph rather than two indexes, a question phrased entirely in Slack's words can
be answered from a GitHub issue — and the answer comes back with the threads and
issues it was grounded in.

## The demo, in one question

Ask this scoped to Slack only:

```
/meridian-ask slack: why did we pause the Neon migration?
```

Slack contains the question being asked, the frustration, and the fact that it
was paused. It does not contain the reason — the team explicitly said "Priya
said she'd write it up properly rather than dribble it out in Slack." So a
Slack-only memory can only tell you that nobody knows.

Now drop the scope:

```
/meridian-ask why did we pause the Neon migration?
```

Now it answers: PgBouncer transaction-mode pooling breaks against Neon's
connection proxy for ~15% of query paths, and branch computes do not scale to
zero, so the projection went from $400 to $1,900/month — 4.75x over the approved
threshold. That reasoning exists only in GitHub issue #412.

No keyword search over Slack can produce that answer, because the words are not
in Slack. That is the entire point of the project.

## Setup

```bash
uv venv --python 3.13 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# The Qdrant adapter pins an exact cognee== version that fights the dev-branch
# pin in requirements.txt, so it installs separately without deps — and then
# needs its own client library, which --no-deps skipped.
VIRTUAL_ENV=.venv uv pip install --no-deps cognee-community-vector-adapter-qdrant==0.4.0
VIRTUAL_ENV=.venv uv pip install "qdrant-client>=1.9"

cp .env.example .env      # then fill in LLM_API_KEY
docker run -d -p 6333:6333 -v "$(pwd)/.qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```

Then build the memory and check it works without involving Slack at all:

```bash
.venv/bin/python ingest.py          # ~1 min on a paid key
.venv/bin/python ask.py --demo      # the scripted demo questions
```

### Slack

```bash
ngrok http 8000 --domain=<your-reserved-domain>   # free tier includes one
```

Put the host into **both** command URLs in `slack-manifest.yaml`, then
api.slack.com/apps → Create New App → From an app manifest → paste. Install to
the workspace, copy the **Signing Secret** from Basic Information into `.env`,
then:

```bash
.venv/bin/python app.py             # :8000
```

## Commands

| Command | What it does |
|---|---|
| `/meridian-ask <question>` | Ask across every source |
| `/meridian-ask slack: <question>` | Scope retrieval to Slack only (the "before" state) |
| `/meridian-ask github: <question>` | Scope to GitHub only |
| `/meridian-remember <fact>` | Record something Slack never saw, folded into the graph immediately |

## How it works

```
Slack export (JSON)  ─┐
                      ├─→ one document per THREAD ─→ cognee.add(node_set=[...])
GitHub issues (JSON) ─┘        + provenance                      │
                                 stamped in                      ▼
                                                          cognee.cognify()
                                                                 │
                                              entities + relationships in the graph
                                                    embeddings in Qdrant
                                                                 │
   /meridian-ask ──→ router picks a SearchType ──→ cognee.recall() ──→ answer
                     (see query.py)              include_references=True   + citations
```

Four decisions carry most of the weight:

**One thread is one document.** A single Slack message is usually meaningless
alone ("Paused or dead?"); the thread is the unit that carries a claim. It also
cuts extraction LLM calls ~5x versus per-message chunking.

**Provenance is written into the document text, not metadata.** cognee's graph
extraction is an LLM reading the document body. Slack's `ts` epoch and channel
name live in the JSON envelope, which the LLM never sees. So every document
opens with `Slack thread in #eng, Thursday 19 March 2026. Participants: ...`.
Skip this and you get a graph that cannot attribute anything to anyone or place
it in time — and every time-scoped question silently degrades to a generic one.

**One dataset, separated by NodeSet.** Both sources land in the `meridian`
dataset tagged `node_set=["slack"|"github", ...]`. Two *datasets* would isolate
them and make cross-source questions impossible; NodeSets keep one graph while
still letting `recall(node_name=["slack"])` reproduce the Slack-only "before"
state on demand. That toggle is the demo.

**The query router.** cognee's search types span ~50x in latency, from ~10ms
lexical chunk search to ~200s chain-of-thought. `query.py` routes per question:
time references → `TEMPORAL`, specific wording or an identifier →
`HYBRID_COMPLETION`, compound questions → `GRAPH_COMPLETION_DECOMPOSITION`,
otherwise `GRAPH_COMPLETION`. Routing is regex, not an LLM call — at this corpus
size a classification round trip would cost more latency than it saves, and a
deterministic router demos the same way twice. `python ask.py --explain "..."`
shows the decision without asking anything.

## Grounding, instead of guessing at refusals

cognee's own Slack integration has an acknowledged problem here: its
`handle_cognee_ask.py` carries `_REFUSAL_MARKERS`, a substring blacklist of
phrasings like `"i can't"` and `"no relevant information"`, to catch answers
where the model didn't actually know. Its docs call this "a whack-a-mole list,
not a structural fix."

We keep that check but demote it to one signal of two. An answer is reported as
**ungrounded** when it either reads like a refusal *or* came back with no
supporting graph references at all (`recall(include_references=True)`). The
second condition is structural: it asks whether retrieval found evidence, not
whether the prose sounds confident. Answers arrive with a `Grounded in:` footer
listing the threads and issues behind them, and ungrounded ones are labelled in
the reply rather than presented as recollection.

## Files

| File | Purpose |
|---|---|
| `memory.py` | cognee wiring: Qdrant adapter registration, dataset and NodeSet names |
| `ingest.py` | Slack export + GitHub issue parsing → documents → graph |
| `query.py` | Search-type routing, grounding checks, citation extraction |
| `ask.py` | CLI access to the same memory. Demo insurance if Slack misbehaves |
| `app.py` | FastAPI Slack endpoint: signature check, 3s ack, background answer |
| `visualize.py` | `graph.html` + per-node-type counts |
| `corpus/` | Seeded Slack export and GitHub issues (see below) |

## About the corpus

`corpus/` is a **seeded fixture, not a real workspace export** — ~28 threads
across 4 channels plus 6 GitHub issues, hand-written so the cross-source
questions have verifiable answers.

It is laid out in genuine Slack export format (a directory per channel, one JSON
file per day, `users.json` and `channels.json` at the root, replies linked by
`thread_ts`), so `ingest.py` parses a real workspace export unchanged. The
constraint on real data is cost, not code: `cognify` is an LLM pass per chunk,
so a full workspace export is tens of minutes of ingest.

## Gotchas found the hard way

- **The Qdrant adapter's `register()` is not a function.** The cognee docs say to
  call `register()`. In adapter 0.4.0 `register` is a *submodule* whose import
  performs the registration; calling it raises `TypeError: 'module' object is not
  callable`. See `memory.py`.
- **`--no-deps` on the adapter also skips `qdrant-client`.** Install it yourself
  or the first cognee call dies with `ModuleNotFoundError: No module named
  'qdrant_client'`.
- **Set `EMBEDDING_PROVIDER=fastembed`.** cognee defaults embeddings to OpenAI
  and reuses `LLM_API_KEY` as the embedding key — with a non-OpenAI key you get
  `Incorrect API key provided: sk-ant-...` *from OpenAI*, which is a baffling
  error to trace.
- **Gemini's free tier is 5 requests/minute**, while cognee's auto-throttle
  guesses 60/min. If you use a Gemini key, set `LLM_RATE_LIMIT_ENABLED=true` and
  `LLM_RATE_LIMIT_REQUESTS=4` or every chunk burns its retry budget on 429s.
- **`ENABLE_BACKEND_ACCESS_CONTROL=false` is required.** Qdrant's dataset handler
  does not support cognee's per-user multi-tenant mode.
- **Slash commands are namespaced `meridian-`, not `cognee-`.** Slack lets
  multiple apps claim the same command and shows a disambiguation dropdown, so at
  an event where many teams install from the same manifest your `/cognee-ask` can
  silently reach someone else's laptop.
- **Stop the server before deleting a dataset.** A running cognee backend holds
  an exclusive lock on the graph database via a worker subprocess.
