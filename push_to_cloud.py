"""Upload the locally-built graph to Cognee Cloud.

    export COGNEE_SERVICE_URL="https://<tenant>.aws.cognee.ai"
    export COGNEE_API_KEY="..."
    python push_to_cloud.py

Why push instead of building in the cloud:

`push()` exports the graph we already built locally as a COGX archive and imports
it remotely, so the cloud does little or no LLM work. `sync()` would ship the raw
files and make the remote instance re-derive everything — paying for extraction
twice and producing a second, subtly different graph.

Keeping the build local also keeps **Qdrant** in the picture, which Cognee Cloud
would otherwise hide behind its own managed storage. Local is the source of
truth; the cloud copy is for sharing and for browsing the graph in a UI.

Note the version skew: this tenant reports cognee 1.4.2 while the local SDK is
1.5.0.dev1. The COGX round trip works across that gap today, but it is the first
thing to suspect if a push starts failing.
"""

import asyncio
import os
import sys

from memory import DATASET, setup


async def main() -> None:
    setup()
    import cognee

    url = os.environ.get("COGNEE_SERVICE_URL")
    key = os.environ.get("COGNEE_API_KEY")
    if not (url and key):
        sys.exit(
            "Set COGNEE_SERVICE_URL and COGNEE_API_KEY first.\n"
            "(cognee also accepts a saved `cognee.serve()` login, but explicit "
            "env vars are easier to reason about.)"
        )

    print(f"pushing dataset '{DATASET}' -> {url}")
    result = await cognee.push(dataset=DATASET, target_dataset=DATASET)

    # PushResult's shape is not stable across runs — an incremental push omits
    # `nodes`/`edges` that a first full push includes. Report whatever came back
    # rather than crashing after the upload has already happened.
    fields = getattr(result, "__dict__", None) or {}
    if not fields:
        fields = {k: getattr(result, k) for k in ("status", "dataset", "target")
                  if hasattr(result, k)}
    print(" ".join(f"{k}={v}" for k, v in fields.items() if v is not None))
    print(
        "\nThe import runs asynchronously on the remote side — 'started' means "
        "accepted, not finished. Confirm with:\n"
        f'  curl -sL -H "X-Api-Key: $COGNEE_API_KEY" '
        f'"$COGNEE_SERVICE_URL/api/v1/datasets/" '
    )


if __name__ == "__main__":
    asyncio.run(main())
