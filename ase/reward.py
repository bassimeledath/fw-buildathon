"""Single-step training reward. Gates multiply, credits add, bounded in [0, 1]. Pure code, no judge."""

from __future__ import annotations

from .env import RESULTS_K, RETRIEVER, parse_action
from .world import FIXED
from .world import SLOTS
REASONS = {"physical_damage", "safety", "billing", "human_request"}
MAX_QUERY_WORDS = 20  # anti-shotgun gate
WRONG_TYPE_CREDIT = 0.1  # valid, harmless, but not the right move


def gate_structure(a: dict | None) -> bool:
    if not a:
        return False
    kind = a.get("action")
    if kind == "search" and FIXED:
        return False
    if kind == "search":
        q = str(a.get("query", "")).split()
        return 0 < len(q) <= MAX_QUERY_WORDS
    if kind == "ask":
        return a.get("slot") in SLOTS and bool(str(a.get("question", "")).strip())
    if kind == "resolve":
        return bool(str(a.get("doc_id", "")).strip())
    if kind == "escalate":
        return a.get("reason") in REASONS
    return False


def score(completion: str, gt: dict) -> dict:
    """Returns {'reward': float, 'kind': str, 'why': str}. gt comes from env.ground_truth()."""
    out = _score(completion, gt)
    if FIXED:  # the simple baseline: full credit or nothing
        out["reward"] = 1.0 if out["reward"] >= 0.99 else 0.0
    return out


def _score(completion: str, gt: dict) -> dict:
    a = parse_action(completion)
    if not gate_structure(a):
        return {"reward": 0.0, "kind": "invalid", "why": "unparseable or malformed action"}
    kind, allowed = a["action"], gt["allowed"]

    # Harmful moves score zero, never partial credit: a wrong fix, or escalating a fixable case.
    if kind == "resolve" and "resolve" not in allowed:
        return {"reward": 0.0, "kind": kind, "why": "resolved when it should not have"}
    if kind == "escalate" and "escalate" not in allowed:
        return {"reward": 0.0, "kind": kind, "why": "escalated a fixable case"}
    if "escalate" in allowed and kind != "escalate":
        return {"reward": 0.0, "kind": kind, "why": "troubleshot a must-escalate case"}
    if kind not in allowed:
        return {"reward": WRONG_TYPE_CREDIT, "kind": kind, "why": f"expected one of {sorted(allowed)}"}

    spec = allowed[kind]
    if kind == "escalate":
        ok = a["reason"] == spec["reason"]
        return {"reward": 1.0 if ok else 0.7, "kind": kind, "why": "reason ok" if ok else "wrong reason"}
    if kind == "resolve":
        ok = a["doc_id"] == spec["doc_id"]
        return {"reward": 1.0 if ok else 0.0, "kind": kind, "why": "gold doc" if ok else "wrong doc"}
    if kind == "ask":
        ok = a["slot"] in spec["slots"]
        return {"reward": 1.0 if ok else 0.2, "kind": kind, "why": "useful slot" if ok else "slot does not split"}
    # search: graded by where the query lands the right article(s) in the frozen BM25 index
    hits = RETRIEVER.search(a["query"], k=RESULTS_K)
    ids = [h.doc_id for h in hits]
    if len(spec["candidates"]) == 1:
        # credit for finding the right situation; the tier (basic vs advanced) is scored at resolve
        cell = {d.doc_id: (d.platform, d.symptom, d.trigger) for d in RETRIEVER.kb}
        hit = [i for i, d in enumerate(ids) if d == spec["gold"] or (d in cell and cell[d] == cell[spec["gold"]]
                                                                     and cell[d][1] != "other")]
        rank = hit[0] + 1 if hit else None
        r = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.6}.get(rank, 0.2)
        return {"reward": r, "kind": kind, "why": f"gold rank {rank}"}
    # ambiguous case: a search is fine if it surfaces the right family of articles
    covered = len(set(ids) & set(spec["candidates"]))
    r = 1.0 if covered >= 2 else (0.6 if covered == 1 else 0.2)
    return {"reward": r, "kind": kind, "why": f"{covered} candidates in top-{RESULTS_K}"}
