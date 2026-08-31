"""decide_action — AIDE-style greedy best-first search with an explicit
"which branch to expand" decision (AI-Scientist-v2's experiment-manager idea).

Returns (action, target_node_or_None, reason). The reason string is logged into
the resulting Node so judges can follow every branching decision.

Four moves. draft / debug / improve are unchanged. `crossover` is the escalation
when the search runs out of single-lineage room: it combines the menu choices of
two distinct successful nodes into one new configuration, which single-lineage
extension can never reach.

Escalation order when the global best stops improving:
    extend best  ->  branch to another live lineage  ->  crossover two lineages
"""
from __future__ import annotations

from .contracts import ExperimentTree, Node

MIN_DRAFTS = 4          # function default; real runs pass draft_count from config
MAX_DEBUG_CHAIN = 2     # give up on a lineage after this many failed debugs
STALL_WINDOW = 3        # improve attempts on one node before branching elsewhere


def _debug_chain_len(tree: ExperimentTree, node: Node) -> int:
    n, length = node, 0
    while n is not None and n.action == "debug":
        length += 1
        n = tree.get(n.parent_id)
    return length


def _is_stalled(tree: ExperimentTree, node: Node) -> bool:
    """True when node's last STALL_WINDOW improve children all failed to beat it."""
    if node.metrics is None:
        return False
    kids = [n for n in tree.children_of(node.iteration_id) if n.action == "improve"]
    recent = kids[-STALL_WINDOW:]
    if len(recent) < STALL_WINDOW:
        return False
    return all((n.status == "error")
               or (n.metrics or {}).get("primary", -1) <= node.metrics["primary"]
               for n in recent)


def crossover_partner(tree: ExperimentTree, primary: Node) -> Node | None:
    """Deterministic second parent for a crossover.

    The highest-scoring successful node whose menu_choices differ from `primary`'s.
    Deterministic so decide_action and build_prompt independently agree on it
    without widening decide_action's return signature.
    """
    if primary is None:
        return None
    for n in sorted(tree.successes(),
                    key=lambda n: (-n.metrics["primary"], n.iteration_id)):
        if n.iteration_id != primary.iteration_id and n.menu_choices != primary.menu_choices:
            return n
    return None


def _families(tree: ExperimentTree) -> dict:
    """Group successful nodes by menu-choice signature -> best node of that family.

    A "lineage" is a configuration family, not a single node. Judging exhaustion
    node-by-node would let an under-performing child of a dead lineage look like a
    fresh direction just because nothing has been tried on top of it yet.
    """
    fams: dict = {}
    for n in tree.successes():
        sig = tuple(sorted(n.menu_choices.items()))
        cur = fams.get(sig)
        if cur is None or n.metrics["primary"] > cur.metrics["primary"]:
            fams[sig] = n
    return fams


def _distinct_configs(tree: ExperimentTree) -> int:
    return len(_families(tree))


def decide_action(tree: ExperimentTree, draft_count: int = MIN_DRAFTS,
                  allow_crossover: bool = True) -> tuple[str, Node | None, str]:
    nodes = tree.nodes

    # 1) most recent attempt errored -> debug it (unless its lineage is hopeless)
    if nodes and nodes[-1].status == "error":
        last = nodes[-1]
        # Some failures invalidate the execution strategy rather than the
        # implementation. Retrying a timed-out script unchanged is not repair;
        # it spends the same budget to learn the same fact. Start a materially
        # different draft and leave the failed artifact out of the lineage.
        from .failure import classify
        failure = classify(last.error_trace)
        if failure.get("needs_shrink") and not failure.get("retry_worthwhile"):
            return ("draft", None,
                    f"pivoting away from node {last.iteration_id}: "
                    f"{failure['class']} requires a materially cheaper "
                    "experiment, not an unchanged debug retry")
        chain = _debug_chain_len(tree, last)
        if chain < MAX_DEBUG_CHAIN:
            return ("debug", last,
                    f"last attempt (node {last.iteration_id}) errored; "
                    f"debug chain length {chain} < {MAX_DEBUG_CHAIN}, feeding the "
                    f"error trace back to fix it")
        reason_prefix = (f"abandoning node {last.iteration_id}: its debug chain "
                         f"already failed {chain} times; ")
    else:
        reason_prefix = ""

    # 2) not enough diverse drafts yet -> draft a fresh high-priority combination
    if len(nodes) < draft_count:
        return ("draft", None,
                reason_prefix + f"only {len(nodes)} attempts exist "
                f"(< draft_count={draft_count}); drafting a fresh combination, "
                f"prioritizing untried high-priority axes")

    best = tree.best()
    if best is None:
        return ("draft", None,
                reason_prefix + "no successful node yet after "
                f"{len(nodes)} attempts; drafting again from the menu")

    # 3) improve — decide WHICH node to expand
    if _is_stalled(tree, best):
        # 3a) branch to the best *other* lineage that is not itself exhausted
        alt = None
        for n in sorted(_families(tree).values(), key=lambda n: -n.metrics["primary"]):
            if n.iteration_id == best.iteration_id or n.menu_choices == best.menu_choices:
                continue
            if not _is_stalled(tree, n):
                alt = n
                break
        if alt is not None:
            return ("improve", alt,
                    reason_prefix + f"best node {best.iteration_id} "
                    f"(primary {best.metrics['primary']:.4f}) looks exhausted — its "
                    f"last {STALL_WINDOW} improve children failed to beat it; branching "
                    f"from node {alt.iteration_id} "
                    f"(primary {alt.metrics['primary']:.4f}, different menu choices) "
                    f"to explore a different neighborhood")

        # 3b) every live lineage is exhausted -> combine two of them
        partner = crossover_partner(tree, best)
        if allow_crossover and partner is not None and _distinct_configs(tree) >= 2:
            return ("crossover", best,
                    reason_prefix + f"every single-lineage direction is exhausted "
                    f"(best node {best.iteration_id} at "
                    f"{best.metrics['primary']:.4f} and every alternative lineage "
                    f"have all stalled); crossing node {best.iteration_id} with node "
                    f"{partner.iteration_id} "
                    f"(primary {partner.metrics['primary']:.4f}) to reach a "
                    f"configuration neither lineage can produce by extension alone")

    return ("improve", best,
            reason_prefix + f"extending global best node {best.iteration_id} "
            f"(valid primary {best.metrics['primary']:.4f}); its neighborhood is "
            f"not exhausted")
