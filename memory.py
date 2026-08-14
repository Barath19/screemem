"""Shared cognee wiring for the Meridian memory bot.

Every entrypoint (ingest.py, ask.py, app.py) imports `setup()` from here and calls
it before touching cognee. Two things must happen in this order:

  1. Registering the Qdrant community adapter, which is what teaches core cognee
     that "qdrant" is a valid vector_db_provider. Registration lives in process
     memory, so every process must do it before its first cognee call.

     Note: the cognee docs tell you to call `register()`. That is wrong for
     adapter 0.4.0 — here `register` is a *submodule* whose import has the
     side effect of registering, so calling it raises
     "TypeError: 'module' object is not callable". Importing the submodule is
     the whole operation.
  2. `load_dotenv()`, because cognee reads its config from the environment at
     first use.

Skipping step 1 gets you:
    OSError: Unsupported vector database provider: qdrant.
"""

import os

from dotenv import load_dotenv

# The dataset every Slack-origin and GitHub-origin document lands in. One
# dataset, separated by NodeSet rather than by dataset, so the graph can relate
# a Slack thread to a GitHub issue. Splitting them into two datasets would make
# the cross-source questions — the entire point of this project — impossible.
DATASET = "meridian"

# NodeSet tags. These become first-class nodes in the graph with belongs_to_set
# edges, and can be used to scope retrieval via recall(node_name=[...]).
SOURCE_SLACK = "slack"
SOURCE_GITHUB = "github"

_READY = False


def setup() -> None:
    """Idempotent. Safe to call from every entrypoint."""
    global _READY
    if _READY:
        return

    load_dotenv()

    # Import-for-side-effect. Do not "clean this up" into a call.
    import cognee_community_vector_adapter_qdrant.register  # noqa: F401

    _READY = True


def require_llm_key() -> None:
    """Fail loudly and early rather than deep inside a pipeline run."""
    if not os.environ.get("LLM_API_KEY"):
        raise SystemExit(
            "LLM_API_KEY is not set.\n"
            "Copy .env.example to .env and put an Anthropic API key in LLM_API_KEY.\n"
            "Then re-run. cognee needs it to extract the knowledge graph."
        )
