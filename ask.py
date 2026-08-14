"""CLI access to the same memory the Slack bot uses.

    python ask.py "why did we pause the Neon migration?"
    python ask.py --source slack "why did we pause the Neon migration?"
    python ask.py --demo

This exists as demo insurance. If ngrok, the tunnel, or Slack's slash-command
routing misbehaves at pitch time, the memory layer -- which is the actual
project -- is still demonstrable from a terminal in one command.
"""

import argparse
import asyncio
import logging
import sys
import warnings

# Cosmetic only: the Qdrant client warns about "Api key is used with an insecure
# connection" against a local http:// instance, and cognee logs its config
# banner at import. Neither is interesting while demoing.
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("cognee").setLevel(logging.ERROR)

from memory import require_llm_key, setup  # noqa: E402
from query import RetrievalFailed, ask, route  # noqa: E402

# The scripted demo. Question 1 is the money shot: the answer exists only in
# GitHub, while every word of the question matches Slack. Running it twice,
# once scoped to Slack and once across both sources, is the whole pitch.
DEMO = [
    ("why did we pause the Neon migration?", "slack"),
    ("why did we pause the Neon migration?", None),
    ("who owns the rate limiter and what is blocking it?", None),
    ("what did we decide about the driver app framework, and did that change?", None),
]

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def render(a) -> str:
    scope = f" [scope: {a.scope}]" if a.scope else ""
    flag = "" if a.grounded else f"  {DIM}(ungrounded — no supporting graph evidence){RESET}"
    out = [
        f"\n{BOLD}Q{RESET} {a.question}{scope}",
        f"{DIM}   routed to {a.search_type} in {a.seconds:.1f}s{RESET}{flag}",
        "",
        a.text,
    ]
    if a.references:
        out.append(f"\n{DIM}   sources:{RESET}")
        out += [f"{DIM}   - {r}{RESET}" for r in a.references]
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="the question to ask")
    ap.add_argument("--source", choices=["slack", "github"], help="scope to one source")
    ap.add_argument("--type", dest="force_type", help="override the routed SearchType")
    ap.add_argument("--demo", action="store_true", help="run the scripted demo questions")
    ap.add_argument("--explain", action="store_true", help="show routing only, ask nothing")
    args = ap.parse_args()

    question = " ".join(args.question).strip()

    if args.explain:
        if not question:
            sys.exit("nothing to explain — pass a question")
        t, reason = route(question)
        print(f"{question}\n  -> {t}  ({reason})")
        return

    setup()
    require_llm_key()

    try:
        if args.demo:
            for q, src in DEMO:
                print(render(await ask(q, source=src)))
                print()
            return

        if not question:
            sys.exit('usage: python ask.py "your question"   (or --demo)')

        print(render(await ask(question, source=args.source, force_type=args.force_type)))
    except RetrievalFailed as exc:
        sys.exit(f"\nretrieval failed:\n{exc}")


if __name__ == "__main__":
    asyncio.run(main())
