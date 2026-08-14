# 2-minute pitch

Judging is 20 points: **runs and is ready to use** (5), **depth not breadth** (0–5),
**complexity — subagents, tooling** (0–5), **novel application** (0–5). Top 5 pitch
to the room, audience votes on Slido. This script is written to hit those four lines
in order, and to survive a demo failure.

---

## 0:00 — the problem, in one sentence

> "Your team's decisions are in Slack. The *reasons* for them usually aren't."

Don't explain knowledge graphs. Don't say "AI memory layer." Everyone before you
has said that. Lead with the thing the room recognises from their own week.

## 0:15 — the setup, out loud, before you type

> "I'm going to ask our Slack memory a question that is *entirely* in Slack's
> words. And it's going to tell me nobody knows."

Naming the outcome in advance turns the first result from a failure into a
prediction you just made. That is what makes the second half land.

## 0:25 — run it, scoped to Slack

```
/meridian-ask slack: why did we pause the Neon migration?
```

> "Priya pulled the branch-per-PR job out of CI. No reason posted."

Point at the citations: three Slack threads. Then say the honest thing:

> "That's correct. The team literally said 'Priya will write it up properly
> rather than dribble it out in Slack.' A keyword search over Slack cannot
> answer this, because the answer is not in Slack."

## 0:50 — drop the scope

```
/meridian-ask why did we pause the Neon migration?
```

> "PgBouncer transaction-mode pooling breaks against Neon's connection proxy.
> And branch computes don't scale to zero — the projection went from $400 to
> $1,900 a month, 4.75x over what was approved."

Then point at the new line in the citations:

> "**GitHub issue 412.** Same question. Same words. Different source. One graph."

**Stop talking for a beat here.** This is the moment the pitch either lands or
doesn't; let the room read the citation themselves.

## 1:10 — depth: why it's one graph and not two indexes

> "Both sources are in one cognee dataset, separated by NodeSet — not two
> datasets. Two datasets can't relate a Slack thread to an issue. One graph can,
> and I can still scope back to Slack-only on demand, which is what you just saw."

One sentence on the non-obvious engineering, because it's the thing nobody else
will have hit:

> "Provenance goes in the document *text*, not metadata. Extraction is an LLM
> reading the body — Slack's timestamp and channel live in the JSON envelope,
> which the LLM never sees. Without that, the graph can't attribute anything to
> anyone or place it in time."

## 1:30 — complexity: the router

> "cognee's search types span 50x in latency — 10 milliseconds for lexical, 200
> seconds for chain-of-thought. So we route per question: time references go to
> TEMPORAL, specific wording goes to HYBRID, everything else to GRAPH_COMPLETION.
> Deterministic, so it demos the same way twice."

## 1:40 — novel: the grounding claim (the strongest 20 seconds)

> "cognee's own Slack integration has a list of phrases like 'I can't' and 'no
> relevant information' to catch answers where the model didn't actually know.
> Their docs call it 'whack-a-mole, not a structural fix.'
>
> We ask a different question: did retrieval find any evidence at all? No
> evidence, no recollection — we label it a guess.
>
> And we found out today why that matters. We pushed this graph to Cognee Cloud.
> The graph transferred; the vector index didn't. So we asked the cloud copy the
> same question, and it told us the migration was paused for 'data-integrity
> errors and latency spikes.' That sentence is nowhere in our data. Fluent,
> confident, invented. Our bot flags exactly that case."

## 2:00 — land it

> "Slack remembers what was said. This remembers why."

---

## If the demo breaks

Do not debug on stage. Two fallbacks, in order:

1. **CLI** — same memory, same code path, no Slack, no tunnel:
   ```bash
   .venv/bin/python ask.py --demo
   ```
   Stop `app.py` first — a running server holds an exclusive lock on the graph
   store and the CLI will refuse with a message telling you so.

2. **The graph** — open `graph.html` and talk over it. 170 nodes, 31 types, 1414
   edges. Point at a Slack thread node and a GitHub issue node sharing an entity.

Say "let me show you the same thing from the terminal" and keep moving. Nobody
deducts points for a tunnel; they deduct for watching you fix one.

## Answers to the likely questions

**"Is this real data?"**
> "No — it's a seeded fixture, and I'd rather say so than have you find out. It's
> in genuine Slack export format, channel directories and thread_ts, so the
> ingester runs against a real export unchanged. The constraint on real data is
> cognify cost, not code: it's an LLM pass per chunk."

**"Why not just use Cognee Cloud?"**
> "Cloud manages its own storage, which hides Qdrant — half of what tonight is
> about. And measured: push carries the graph but not the vector index, so the
> cloud copy answers from nothing. We build locally against Qdrant and treat the
> cloud copy as transport."

**"What would you do next?"**
> "Custom graph model — Person, Thread, Decision as typed nodes with a supersedes
> edge — so 'what did we decide, and what did we decide before that' is a graph
> traversal instead of something the LLM has to infer from two documents."

**"How long did this take?"**
> Be straight. The number is less interesting than what you cut: no OAuth, no
> per-user memory, no chain-of-thought retrieval. Depth over breadth is the
> rubric line, so say what you deliberately left out.
