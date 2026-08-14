"""A multi-agent query pipeline over the same cognee graph.

    planner  ->  N parallel retrievers  ->  verifier  ->  synthesiser

The fast path in `query.py` is one retrieval and one completion. That is the
right default. This module exists for questions where a single retrieval sweep
genuinely cannot see the whole answer, and it is built to strengthen the one
claim this project actually makes — that an answer should be traceable to
evidence — rather than to add stages for their own sake.

What each agent is for:

**Planner.** Decomposes the question into sub-questions and, per sub-question,
picks which source to look in. This matters because our two sources hold
different *kinds* of content: Slack holds who-said-what-and-when, GitHub holds
the reasoning. "Who owns X and why is it blocked" is two lookups against two
sources, and one blended retrieval tends to return the louder one.

**Retrievers, in parallel.** One `cognee.recall` per plan step, each scoped to
its source via `node_name`, all fired concurrently with `asyncio.gather`. Wall
clock is the slowest single retrieval, not their sum.

**Verifier.** The adversarial stage, and the reason this pipeline earns its
latency. It receives the draft claims and the retrieved evidence *separately*
and is instructed to mark a claim unsupported unless the evidence states it.
Document-level grounding — the fast path's check — proves retrieval found
something. Claim-level verification asks whether it found *this*. An answer can
cite three real threads and still contain one invented sentence; only this stage
catches that.

**Synthesiser.** Merges surviving claims into one answer, and reports what it
dropped instead of quietly omitting it.

Every agent runs through cognee's own `LLMGateway.acreate_structured_output`, so
they inherit the configured provider and model and add no new dependency.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from memory import DATASET, SOURCE_GITHUB, SOURCE_SLACK
from query import extract_references, extract_text

# --------------------------------------------------------------------------
# Agent I/O schemas. These are the structured-output contracts, so the LLM
# cannot return a shape the pipeline does not expect.
# --------------------------------------------------------------------------


class PlanStep(BaseModel):
    sub_question: str = Field(..., description="A single self-contained question.")
    source: Literal["slack", "github", "both"] = Field(
        ...,
        description=(
            "Where to look. Use 'slack' for who said what, when, and team "
            "sentiment. Use 'github' for reasons, decisions, blockers and "
            "technical detail. Use 'both' only when genuinely unclear."
        ),
    )
    why: str = Field(..., description="One short clause on why this source.")


class Plan(BaseModel):
    subject: str = Field(
        ...,
        description=(
            "The single main thing the question is about, as a short noun phrase "
            "using the question's own words — e.g. 'Neon migration', 'rate "
            "limiter', 'driver app framework'. Not a sentence. This is checked "
            "against the retrieved evidence to detect questions about things the "
            "memory has never heard of."
        ),
    )
    steps: list[PlanStep] = Field(
        ..., description="Between 1 and 4 steps. Prefer the fewest that cover the question."
    )


class ClaimCheck(BaseModel):
    claim: str = Field(..., description="A single factual assertion from the draft.")
    supported: bool = Field(
        ..., description="True only if the evidence explicitly states this claim."
    )
    evidence_quote: str = Field(
        "", description="Short quote from the evidence supporting it, or empty if unsupported."
    )


class Verification(BaseModel):
    claims: list[ClaimCheck]


class Synthesis(BaseModel):
    answer: str = Field(..., description="Final answer using only supported claims.")
    dropped: list[str] = Field(
        default_factory=list, description="Claims removed for lack of support."
    )


@dataclass
class AgentAnswer:
    question: str
    text: str
    seconds: float
    subject: str = ""
    plan: list[PlanStep] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    supported: int = 0
    unsupported: int = 0
    dropped: list[str] = field(default_factory=list)
    grounded: bool = True


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------

PLANNER_PROMPT = """You plan retrieval over a team's memory built from two sources.

SLACK holds conversation: who said what, when, who to ask, team sentiment, and
what people did not know at the time.
GITHUB holds issues: decisions, the reasoning behind them, blockers, owners,
costs, and technical detail. Reasons usually live here, not in Slack.

Break the user's question into the fewest self-contained sub-questions that
together answer it, and pick the source for each. If the question asks both
"who/when" and "why", that is two steps against two different sources.

Use 'both' for ownership and responsibility questions ("who owns", "who is
working on", "who should I ask"). Ownership appears as a GitHub assignee AND as
someone saying "X is on it" in Slack, and the two disagree often enough that
seeing only one is misleading.

Name the specific subject in each sub-question rather than using a pronoun or a
bare noun. "Why is the rate limiter rewrite blocked" retrieves better than "why
is it blocked", and it stops a later stage from attaching the answer to the wrong
thing when several similar items exist.
Return only the plan."""

VERIFIER_PROMPT = """You are a strict fact-checker. You are given a QUESTION, a
DRAFT answer, and the EVIDENCE that was retrieved from a knowledge graph.

Split the draft into individual factual claims. For each claim decide whether the
EVIDENCE explicitly supports it.

Rules:
- Mark supported=false unless the evidence states the claim. Plausibility is not
  support. Your own world knowledge is not support.
- A claim that is merely consistent with the evidence is NOT supported.
- If supported, quote the specific span of evidence VERBATIM — copy the characters
  exactly, do not paraphrase or reflow. The quote is checked automatically
  against the evidence, and a quote that cannot be found there is treated as no
  support at all.
- WATCH FOR SUBJECT CONFUSION. The evidence describes several similar items
  (multiple issues, multiple projects). A property of one is not a property of
  another. If a claim says item A is unowned/blocked/closed, the quote must be
  about item A specifically, not about a neighbouring item mentioned nearby. This
  is the most common way a wrong claim slips through.
- Absence of information is itself a valid claim: "no reason was given in Slack"
  is supported if the evidence shows discussion without a reason."""

SYNTH_PROMPT = """Write the final answer to the question using ONLY the claims
marked supported. Be direct and concise — two or three sentences. Do not add new
facts, do not hedge unnecessarily, and do not mention the verification process.
If the supported claims do not answer the question, say plainly what is not
known. List any claims you dropped."""


async def plan_question(question: str) -> Plan:
    from cognee.infrastructure.llm import LLMGateway

    return await LLMGateway.acreate_structured_output(question, PLANNER_PROMPT, Plan)


async def retrieve(step: PlanStep) -> tuple[str, str]:
    """One plan step -> (answer text, evidence text). Runs both calls concurrently."""
    import cognee
    from cognee.modules.search.types import SearchType

    node_name = None
    if step.source == SOURCE_SLACK:
        node_name = [SOURCE_SLACK]
    elif step.source == SOURCE_GITHUB:
        node_name = [SOURCE_GITHUB]

    common = dict(
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        top_k=10,
        node_name=node_name,
        auto_route=False,
    )
    answer, context = await asyncio.gather(
        cognee.recall(step.sub_question, **common),
        cognee.recall(step.sub_question, only_context=True, **common),
        return_exceptions=True,
    )

    a = extract_text(answer[0]) if not isinstance(answer, BaseException) and answer else ""
    c = extract_text(context[0]) if not isinstance(context, BaseException) and context else ""
    return a, c


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


# Words that carry no subject identity, so their presence in the evidence proves
# nothing about whether the memory knows the thing being asked about.
_GENERIC = {
    "migration", "migrating", "decision", "decisions", "project", "issue",
    "problem", "change", "update", "upgrade", "plan", "team", "work", "status",
    "the", "our", "about", "from", "with", "that", "this", "have", "make",
}


def subject_is_known(subject: str, evidence: str) -> bool:
    """Does the evidence actually mention the thing the question is about?

    This gate exists because claim-level verification cannot catch a false
    premise. Asked "what did we decide about migrating to Kubernetes", retrieval
    returns the Neon migration documents, the verifier correctly confirms every
    claim about Neon, and the synthesiser then writes "decisions about migrating
    to Kubernetes included pausing the Neon staging migration". Every individual
    claim is true; the sentence is fabricated.

    So: strip the generic words from the subject and require at least one
    identifying word to appear in the evidence. 'Kubernetes' is absent -> refuse.
    'Neon migration' keeps 'neon', which is present -> proceed.
    """
    hay = _norm(evidence)
    words = [w.strip(".,#") for w in _norm(subject).split()]
    identifying = [w for w in words if len(w) > 3 and w not in _GENERIC]
    if not identifying:
        # Nothing distinctive to check (e.g. subject was "the project"); do not
        # block on a test that cannot discriminate.
        return True
    return any(w in hay for w in identifying)


def enforce_quotes(verification: Verification, evidence: str) -> Verification:
    """Demote any 'supported' claim whose quote is not actually in the evidence.

    A verifier that both judges support and reports its own reasoning can simply
    assert support — which is what happened in testing: it passed 5 of 5 claims
    including a false one. Requiring a verbatim quote and then checking that quote
    against the evidence in code turns the verifier's output into something
    falsifiable. It cannot claim support for text that does not exist.

    Matching is whitespace-normalised, and long quotes are accepted on a generous
    prefix so that a trailing paraphrase does not reject an otherwise real quote.
    """
    hay = _norm(evidence)
    checked: list[ClaimCheck] = []
    for c in verification.claims:
        if not c.supported:
            checked.append(c)
            continue

        q = _norm(c.evidence_quote)
        if len(q) < 12:
            # Too short to be evidence of anything; "open" or "marcus" matches
            # everywhere and proves nothing.
            checked.append(c.model_copy(update={"supported": False}))
            continue

        # Verbatim containment only, on the whole quote or a long prefix of it.
        #
        # A lexical-overlap fallback was tried here and removed. Requiring 80% of
        # the quote's content words to appear anywhere in the evidence let through
        # "the decision about migrating to Kubernetes was discussed on 10 February
        # 2026" — pure fabrication — because "decision", "migrating" and
        # "February" all occur in evidence about the *Neon* migration. Bag-of-words
        # scoring cannot distinguish two subjects that share vocabulary, which is
        # the exact confusion this stage exists to catch.
        #
        # Strictness costs completeness: a verifier that paraphrases loses a claim
        # it could have kept. That is the right trade here. A dropped true claim is
        # visible in the `dropped` list and the user can go read the source; an
        # accepted false claim is invisible and is exactly the failure this whole
        # project argues against.
        ok = q in hay or _norm(c.evidence_quote[:120]) in hay
        checked.append(c if ok else c.model_copy(update={"supported": False}))

    return Verification(claims=checked)


async def verify(question: str, draft: str, evidence: str) -> Verification:
    from cognee.infrastructure.llm import LLMGateway

    payload = (
        f"QUESTION:\n{question}\n\n"
        f"DRAFT:\n{draft}\n\n"
        f"EVIDENCE:\n{evidence[:12000]}"
    )
    return await LLMGateway.acreate_structured_output(payload, VERIFIER_PROMPT, Verification)


async def synthesise(question: str, verification: Verification) -> Synthesis:
    from cognee.infrastructure.llm import LLMGateway

    lines = [
        f"[{'SUPPORTED' if c.supported else 'UNSUPPORTED'}] {c.claim}"
        for c in verification.claims
    ]
    payload = f"QUESTION:\n{question}\n\nCLAIMS:\n" + "\n".join(lines)
    return await LLMGateway.acreate_structured_output(payload, SYNTH_PROMPT, Synthesis)


async def ask_with_agents(question: str) -> AgentAnswer:
    import time

    started = time.monotonic()

    plan = await plan_question(question)
    steps = plan.steps[:4] or [PlanStep(sub_question=question, source="both", why="fallback")]

    # All retrievers at once: wall clock is the slowest step, not the sum.
    gathered = await asyncio.gather(*(retrieve(s) for s in steps), return_exceptions=True)

    drafts, evidence_parts = [], []
    for step, result in zip(steps, gathered):
        if isinstance(result, BaseException):
            continue
        a, c = result
        if a:
            drafts.append(f"{step.sub_question}\n{a}")
        if c:
            evidence_parts.append(c)

    evidence = "\n\n".join(evidence_parts)
    refs = extract_references(evidence)

    if not drafts:
        return AgentAnswer(
            question=question,
            text="Nothing in memory covers that.",
            seconds=time.monotonic() - started,
            plan=steps,
            grounded=False,
        )

    # Premise gate, before any synthesis. Retrieval always returns its nearest
    # neighbours, so a question about something absent still comes back with
    # plausible-looking documents about something else.
    if not subject_is_known(plan.subject, evidence):
        return AgentAnswer(
            question=question,
            text=(
                f"Nothing in memory mentions {plan.subject}. "
                f"Retrieval returned the closest documents it had, but none of them "
                f"are about {plan.subject}, so there is nothing here to report."
            ),
            seconds=time.monotonic() - started,
            plan=steps,
            subject=plan.subject,
            references=refs,
            grounded=False,
        )

    verification = await verify(question, "\n\n".join(drafts), evidence)
    verification = enforce_quotes(verification, evidence)
    supported = [c for c in verification.claims if c.supported]
    unsupported = [c for c in verification.claims if not c.supported]

    synthesis = await synthesise(question, verification)

    return AgentAnswer(
        question=question,
        text=synthesis.answer,
        seconds=time.monotonic() - started,
        subject=plan.subject,
        plan=steps,
        references=refs,
        supported=len(supported),
        unsupported=len(unsupported),
        dropped=synthesis.dropped or [c.claim for c in unsupported],
        # Claim-level grounding: evidence must exist AND at least one claim must
        # survive verification. The fast path can only check the first half.
        grounded=bool(refs) and bool(supported),
    )
