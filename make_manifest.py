"""Generate a ready-to-paste Slack manifest from the running ngrok tunnel.

    ngrok http 8000            # in another terminal
    python make_manifest.py    # writes slack-manifest.local.yaml

Why this exists: the single most common failure in this setup is replacing
`<public-host>` in one command URL and not the other. Slack then returns
`dispatch_failed` for that one command only — an asymmetric failure that looks
like a backend bug and costs a debugging session. Reading the host from ngrok's
local API and substituting every occurrence makes that mistake impossible.
"""

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE = HERE / "slack-manifest.yaml"
OUT = HERE / "slack-manifest.local.yaml"
NGROK_API = "http://127.0.0.1:4040/api/tunnels"


def public_host() -> str:
    try:
        with urllib.request.urlopen(NGROK_API, timeout=5) as r:
            tunnels = json.load(r).get("tunnels", [])
    except (urllib.error.URLError, OSError):
        sys.exit(
            "Could not reach ngrok's local API at 127.0.0.1:4040.\n"
            "Start the tunnel first:  ngrok http 8000\n"
            "(If ngrok is running but this still fails, it has not authenticated: "
            "ngrok config add-authtoken <token>)"
        )

    https = [t["public_url"] for t in tunnels if t.get("public_url", "").startswith("https://")]
    if not https:
        sys.exit("ngrok is running but has no https tunnel. Expected `ngrok http 8000`.")
    return https[0].removeprefix("https://")


def main() -> None:
    host = public_host()
    manifest = TEMPLATE.read_text()

    filled = manifest.replace("<public-host>", host)
    if "<public-host>" in filled:
        sys.exit("substitution failed — placeholders remain")

    urls = re.findall(r"url: (\S+)", filled)
    if len({u for u in urls}) != 1:
        print(f"warning: command URLs are not identical: {urls}", file=sys.stderr)

    OUT.write_text(filled)

    print(f"host: {host}")
    print(f"wrote {OUT.name} with {len(urls)} command URL(s), all pointing at that host\n")
    print("Next:")
    print("  1. api.slack.com/apps -> Create New App -> From an app manifest")
    print(f"  2. paste {OUT.name}")
    print("  3. Install to Workspace")
    print("  4. Basic Information -> copy Signing Secret into .env as SLACK_SIGNING_SECRET")
    print("  5. restart app.py (env vars are read at startup)")
    print("\nIf you later edit the manifest, Slack requires REINSTALLING the app")
    print("before a new/changed command reaches your server at all.")


if __name__ == "__main__":
    main()
