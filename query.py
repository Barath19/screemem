"""Query routing and answer formatting, shared by the CLI and the Slack bot.

Why a router exists at all: cognee's search types differ by ~50x in latency,
from ~10ms for lexical chunk search to ~200s for chain-of-thought graph
completion. Sending every question to one type means either answering shallow
questions expensively or deep questions badly. So we pick per question.

The routing is deliberately heuristic (regex over the question) rather than an
LLM classification call. At this corpus size an extra LLM round trip would cost
more latency than it saves, and a deterministic router is something you can
demo the same way twice.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field

from memory import DATASET, SOURCE_GITHUB, SOURCE_SLACK

# Explicit time references. Only these route to TEMPORAL, because TEMPORAL costs
# ~40s (3 LLM calls: extract the time window, filter the graph, answer) and
# silently degrades to ordinary graph completion when it finds no time
# constraint — so firing it speculatively buys a slow no-op.
_TIME = re.compile(
    r"\b(before|after|between|during|when did|as of|since|until|"
    r"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|"
    r"aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?|"
    r"q[1-4]|20\d\d|last (week|month|quarter|year)|originally|initially)\b",
    re.I,
)

# Questions about who said a specific thing, or that quote wording, benefit from
# lexical matching fused with semantics — HYBRID puts lexical chunks, semantic
# chunks and graph entity context all in front of the model.
_LEXICAL = re.compile(r"\b(who said|who mentioned|who asked|exact|quote|ticket \d+|#\d+)\b", re.I)

# Multi-part questions get decomposed into subqueries, each retrieved
# separately, then synthesised. ~60s, so gated on real signals of compoundness.
_COMPOUND = re.compile(r"\b(and also|as well as|compare|difference between|both)\b|\?.*\?", re.I)


class RetrievalFailed(RuntimeError):
    """Retrieval broke. Distinct from 'the memory has no answer'.

    The most common cause by far is the graph-database lock: a running cognee
    process (app.py) holds an exclusive lock on the Ladybug graph store via a
    worker subprocess, so a second process cannot read it concurrently.
    """

    LOCK_HINT = (
        "The graph database is locked by another cognee process.\n"
        "app.py and ask.py cannot run at the same time — a running server holds "
        "an exclusive lock on the graph store.\n"
        "Stop app.py and retry, or ask this question through Slack instead."
    )

    def __str__(self) -> str:
        detail = super().__str__()
        if "Could not set lock" in detail or "Lock is held by" in detail:
            return self.LOCK_HINT
        return detail


@dataclass
class Answer:
    question: str
    text: str
    search_type: str
    seconds: float
    references: list[str] = field(default_factory=list)
    scope: str | None = None
    grounded: bool = True


def route(question: str) -> tuple[str, str]:
    """Returns (SearchType name, one-line reason for the choice)."""
    if _TIME.search(question):
        return "TEMPORAL", "question refers to a point or span in time"
    if _COMPOUND.search(question):
        return "GRAPH_COMPLETION_DECOMPOSITION", "compound question, split into subqueries"
    if _LEXICAL.search(question):
        return "HYBRID_COMPLETION", "asks about specific wording or an identifier"
    return "GRAPH_COMPLETION", "default: semantic seeds expanded through the graph"


# Phrasings that mean "the memory did not actually know this". cognee's own
# Slack integration ships a substring blacklist for these (_REFUSAL_MARKERS in
# handle_cognee_ask.py) and its docs call it "whack-a-mole, not a structural
# fix". We keep the check but treat it as one signal among several rather than
# the whole answer: a reply is reported as ungrounded when it either sounds like
# a refusal OR came back with no supporting graph references.
_REFUSAL = re.compile(
    r"\b(i can'?t|i cannot|i'?m unable to|can'?t answer|cannot answer|"
    r"no information about|no relevant information|does(n'?t| not) contain|"
    r"contains no information|not enough (information|context)|"
    r"is not (mentioned|specified|available))\b",
    re.I,
)


def extract_text(entry) -> str:
    for attr in ("text", "content", "answer", "value"):
        v = getattr(entry, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    if isinstance(entry, dict):
        for k in ("text", "content", "answer", "value"):
            if isinstance(entry.get(k), str) and entry[k].strip():
                return entry[k].strip()
    return str(entry).strip()


# The provenance headers ingest.py stamps at the top of every document. Because
# they are in the document *text*, they survive into the retrieved context, which
# is what makes real citation possible.
#
# Note: recall(include_references=True) exists in cognee 1.5.0.dev1 but populates
# nothing for GRAPH_COMPLETION -- the result's `raw` payload comes back as just
# {"value": "<answer>"}. So we get evidence from a second retrieval with
# only_context=True, which returns the nodes retrieval actually surfaced and
# skips the completion call. It runs concurrently with the answer, so it costs
# no wall-clock latency.
_CITATION = re.compile(
    r"(GitHub issue \S+#\d+: [^\n]{0,90}"
    r"|Slack thread in #[\w-]+, [^\n.]{0,60}"
    # Facts added live via /meridian-remember carry this header instead of a
    # thread header, so they are citable on the same footing as exported history.
    r"|Fact recorded in Slack on [^\n.]{0,60}"
    # Screen observations captured by screen_memory.py. Citable for the same
    # reason: you should be able to see that an answer came from your own screen
    # at a specific time rather than from a teammate's message.
    r"|Screen observation on [^\n.]{0,60})"
)


def extract_references(context_text: str) -> list[str]:
    """Human-readable citations for the documents retrieval actually used."""
    found: list[tuple[str, str]] = []  # (normalised key, display form)
    for m in _CITATION.finditer(context_text or ""):
        ref = m.group(1).strip()
        # Graph node names carry a truncation ellipsis and a bracketed keyword
        # tag ("...pausing the Neon... [branch, neon, open]"). Cut both off so
        # the citation reads like a citation.
        ref = re.split(r"\.\.\.|\s\[|\s\(This chunk|\s*Source:", ref)[0].strip().rstrip(".,")
        found.append((re.sub(r"\W+", "", ref.lower()), ref))

    # The graph stores the same document under several labels: a truncated node
    # name and the full header ("Slack thread in #eng, Thursday 19 March"
    # alongside "...19 March 2026"). The short forms are always prefixes of the
    # long ones, so collapse by prefix — longest first, and drop anything that is
    # a prefix of something already kept. Keying on a fixed-length slice instead
    # would split exactly the pairs it is meant to merge.
    kept: list[tuple[str, str]] = []
    for key, ref in sorted(found, key=lambda kv: len(kv[0]), reverse=True):
        if not any(existing.startswith(key) for existing, _ in kept):
            kept.append((key, ref))
    return [ref for _, ref in kept][:6]


async def ask(question: str, source: str | None = None, force_type: str | None = None) -> Answer:
    """source: None (everything), "slack", or "github" — scopes retrieval to a NodeSet."""
    import cognee
    from cognee.modules.search.types import SearchType

    type_name, _reason = route(question)
    if force_type:
        type_name = force_type.upper()

    node_name = None
    if source in (SOURCE_SLACK, SOURCE_GITHUB):
        node_name = [source]

    common = dict(
        query_type=getattr(SearchType, type_name),
        datasets=[DATASET],
        top_k=10,
        node_name=node_name,
        auto_route=False,  # we route ourselves; don't pay for a second decision
    )

    started = time.monotonic()
    # Answer and evidence in parallel. The evidence call skips the completion
    # step (only_context=True), so it finishes first and adds no latency.
    answer_results, context_results = await asyncio.gather(
        cognee.recall(question, **common),
        cognee.recall(question, only_context=True, **common),
        return_exceptions=True,
    )

    elapsed = time.monotonic() - started

    # An error and an empty memory are different things and must not be reported
    # the same way. Collapsing a failure into "nothing in memory covers that" is
    # precisely the class of bug this project is arguing against: it presents an
    # infrastructure failure as a confident statement about what the team knows.
    if isinstance(answer_results, BaseException):
        raise RetrievalFailed(str(answer_results)) from answer_results

    if not answer_results:
        return Answer(
            question=question,
            text="Nothing in memory covers that.",
            search_type=type_name,
            seconds=elapsed,
            scope=source,
            grounded=False,
        )

    text = extract_text(answer_results[0])

    refs: list[str] = []
    if not isinstance(context_results, BaseException) and context_results:
        refs = extract_references(extract_text(context_results[0]))

    # Two independent grounding signals. The refusal check reads the prose, which
    # is the whack-a-mole approach cognee's own integration admits to; the
    # reference check asks whether retrieval found any evidence at all, which is
    # structural. An answer must pass both to be reported as a recollection.
    grounded = bool(refs) and not _REFUSAL.search(text)

    return Answer(
        question=question,
        text=text,
        search_type=type_name,
        seconds=elapsed,
        references=refs,
        scope=source,
        grounded=grounded,
    )
