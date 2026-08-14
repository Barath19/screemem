"""Render the knowledge graph to an HTML file, and print a text summary of it.

    python visualize.py

Two outputs, both useful in a demo:
  graph.html  — the interactive graph. Open it and point at the edges that
                connect a Slack thread to a GitHub issue; that edge is the
                reason the cross-source answer is possible.
  stdout      — per-type node counts, so you can say "412 nodes, 38 of them
                people" instead of gesturing at a hairball.
"""

import asyncio

from memory import DATASET, setup


async def main() -> None:
    setup()
    import cognee

    out = "graph.html"
    await cognee.visualize_graph(out)
    print(f"wrote {out}")

    try:
        inventory = await cognee.get_schema_inventory(dataset_name=DATASET)
    except Exception:
        try:
            inventory = await cognee.get_schema_inventory()
        except Exception as exc:
            print(f"(schema inventory unavailable: {exc})")
            return

    # get_schema_inventory() returns a list of per-type dicts, each with
    # count / samples / relationships. Printing it raw is unreadable — the
    # relationships lists alone run to thousands of characters.
    if isinstance(inventory, dict):
        rows = [{"type": k, **(v if isinstance(v, dict) else {"count": v})}
                for k, v in inventory.items()]
    elif isinstance(inventory, list):
        rows = inventory
    else:
        print(inventory)
        return

    # Structural node types cognee creates for its own bookkeeping. Separated out
    # so the domain types — the interesting ones — are not buried among them.
    PLUMBING = {"DocumentChunk", "TextDocument", "TextSummary", "NodeSet"}

    domain = [r for r in rows if r.get("type") not in PLUMBING]
    plumbing = [r for r in rows if r.get("type") in PLUMBING]

    def show(title, group):
        print(f"\n{title}")
        print("-" * 62)
        for r in sorted(group, key=lambda r: -(r.get("count") or 0)):
            samples = ", ".join(str(s) for s in (r.get("samples") or [])[:3] if s)
            line = f"{str(r.get('type'))[:26]:<26} {str(r.get('count')):>4}"
            if samples:
                line += f"  {samples[:30]}"
            print(line)

    show("domain node types", domain)
    show("cognee internal node types", plumbing)

    total = sum(r.get("count") or 0 for r in rows)
    edges = sum(
        rel.get("count") or 0
        for r in rows
        for rel in (r.get("relationships") or [])
    )
    print(f"\n{total} nodes across {len(rows)} types, {edges} edge instances")


if __name__ == "__main__":
    asyncio.run(main())
