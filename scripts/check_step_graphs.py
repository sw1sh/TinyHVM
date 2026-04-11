#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

NODE_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*\[(.*)\]\s*;\s*$')
EDGE_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*(?:\[(.*)\])?\s*;\s*$')
LABEL_RE = re.compile(r'label="([^"]*)"')
STEP_RE = re.compile(r'^step_(\d+)(?:_(.*))?\.dot$')
TENSOR_LABEL_RE = re.compile(r'^t\d+$')
RAW_HEAP_NODE_RE = re.compile(r'^h\d+$')
SHAPE_LABEL_RE = re.compile(r'\\n\[([0-9?,]+)\]')
PREV_RE = re.compile(r'^\s*//\s*PREV_INTERACTION:\s*(.+?)\s*$')
NEXT_RE = re.compile(r'^\s*//\s*NEXT_INTERACTION:\s*(.+?)\s*$')

BINARY_OPS = {
    "ADD", "SUB", "MUL", "DIV", "MAX", "CMP", "MM",
    "EQL", "AND", "OR", "MAT"
}
UNARY_OPS = {"NEG", "RELU", "EXP", "LOG", "SQRT"}
VIEW_OPS = {"RESHAPE", "PERMUTE", "EXPAND", "SHRINK", "PAD"}
PASSTHRU_UNARY_OPS = {"LOG_PRINT"}


@dataclass
class DotGraph:
    path: str
    step: int
    suffix: str
    prev_interaction: str = ""
    next_interaction: str = ""
    nodes: Dict[str, str] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)

    def kind(self, node_id: str) -> str:
        lbl = self.nodes.get(node_id, "")
        head = lbl.split("\\n", 1)[0]
        if head == "ERA":
            return "ERA"
        if head == "DUP":
            return "DUP"
        if TENSOR_LABEL_RE.match(head):
            return "TEN"
        if head.startswith("num"):
            return "NUM"
        return "NODE"

    def fingerprint(self) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str, str], ...]]:
        ns = tuple(sorted((nid, self.nodes[nid]) for nid in self.nodes))
        es = tuple(sorted(self.edges))
        return ns, es


def parse_dot(path: str) -> DotGraph:
    name = os.path.basename(path)
    m = STEP_RE.match(name)
    if not m:
        raise ValueError(f"unexpected step filename: {name}")
    step = int(m.group(1))
    suffix = m.group(2) or ""
    g = DotGraph(path=path, step=step, suffix=suffix)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pm = PREV_RE.match(line)
            if pm:
                g.prev_interaction = pm.group(1).strip()
                continue
            xm = NEXT_RE.match(line)
            if xm:
                g.next_interaction = xm.group(1).strip()
                continue
            nm = NODE_RE.match(line)
            if nm:
                nid, attrs = nm.groups()
                if nid in {"node", "edge", "graph"}:
                    continue
                lm = LABEL_RE.search(attrs)
                g.nodes[nid] = lm.group(1) if lm else ""
                continue
            em = EDGE_RE.match(line)
            if em:
                src, dst, attrs = em.groups()
                g.edges.append((src, dst, attrs or ""))
    return g


def collect_graphs(step_dir: str) -> List[DotGraph]:
    paths = sorted(glob.glob(os.path.join(step_dir, "step_*.dot")))
    graphs = [parse_dot(p) for p in paths]
    graphs.sort(key=lambda g: (g.step, g.suffix))
    return graphs


def edge_label(attrs: str) -> str:
    m = LABEL_RE.search(attrs or "")
    if not m:
        return ""
    lbl = m.group(1)
    # normalize "(dp0)/(dp1)" suffixes
    dp = lbl.find(" (dp")
    if dp >= 0:
        lbl = lbl[:dp]
    return lbl.strip()

def edge_dp_port(attrs: str) -> str:
    m = LABEL_RE.search(attrs or "")
    if not m:
        return ""
    lbl = m.group(1)
    if "dp0" in lbl:
        return "dp0"
    if "dp1" in lbl:
        return "dp1"
    return ""


def node_head(g: DotGraph, node_id: str) -> str:
    return g.nodes.get(node_id, "").split("\\n", 1)[0]


def has_erase_path_from_tensor(g: DotGraph, tensor_node: str,
                               out_map: Dict[str, List[Tuple[str, str, str]]]) -> bool:
    q: List[Tuple[str, int]] = [(tensor_node, 0)]
    seen = {tensor_node}
    while q:
        nid, d = q.pop(0)
        if d > 6:
            continue
        for _, dst, _ in out_map.get(nid, []):
            if g.kind(dst) == "ERA":
                return True
            if dst not in seen:
                seen.add(dst)
                q.append((dst, d + 1))
    return False


def prev_interaction_name(g: DotGraph) -> str:
    if g.prev_interaction:
        return g.prev_interaction
    if g.suffix.startswith("state_init"):
        return "state_init"
    if g.suffix and not g.suffix.startswith("state_"):
        return g.suffix
    return ""


def interactions_match(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    dp_steps = {"interact_DP0", "interact_DP1"}
    return expected in dp_steps and actual in dp_steps


def check_graphs(graphs: List[DotGraph]) -> List[str]:
    errs: List[str] = []
    g0 = graphs[0]
    init_grad_count = sum(1 for lbl in g0.nodes.values() if lbl.split("\\n", 1)[0] == "GRAD")
    has_init_grad = init_grad_count > 0
    if not has_init_grad:
        errs.append(f"{os.path.basename(g0.path)}: init graph must contain a GRAD node before any interaction")

    gl = graphs[-1]
    has_final_grad = any(lbl.split("\\n", 1)[0] == "GRAD" for lbl in gl.nodes.values())
    if has_final_grad:
        errs.append(f"{os.path.basename(gl.path)}: phase-1 final graph still contains GRAD nodes")
    if not gl.suffix.startswith("state_final"):
        errs.append(f"{os.path.basename(gl.path)}: phase-1 graph sequence must end with explicit state_final")
    if not prev_interaction_name(g0):
        errs.append(f"{os.path.basename(g0.path)}: missing PREV_INTERACTION metadata")
    if len(graphs) > 1 and not gl.prev_interaction:
        errs.append(f"{os.path.basename(gl.path)}: final graph must record PREV_INTERACTION metadata")

    final_out_map: Dict[str, List[Tuple[str, str, str]]] = {}
    final_adj: Dict[str, List[str]] = {nid: [] for nid in gl.nodes}
    for e in gl.edges:
        final_out_map.setdefault(e[0], []).append(e)
        final_adj[e[0]].append(e[1])
        final_adj[e[1]].append(e[0])
    init_out_map: Dict[str, List[Tuple[str, str, str]]] = {}
    for e in g0.edges:
        init_out_map.setdefault(e[0], []).append(e)
    init_roots = sorted(nid for nid in g0.nodes if not init_out_map.get(nid))
    final_roots = sorted(nid for nid in gl.nodes if not final_out_map.get(nid))
    if gl.nodes:
        min_final_roots = sum(1 for nid in init_roots if node_head(g0, nid) != "GRAD")
        max_final_roots = len(init_roots)
        if len(final_roots) < min_final_roots:
            errs.append(
                f"{os.path.basename(gl.path)}: final graph has {len(final_roots)} result roots "
                f"(expected at least {min_final_roots} non-GRAD roots from init graph to remain live)"
            )
        if len(final_roots) > max_final_roots:
            errs.append(
                f"{os.path.basename(gl.path)}: final graph has {len(final_roots)} result roots "
                f"(expected no more than {max_final_roots} init roots)"
            )
        if not final_roots:
            errs.append(f"{os.path.basename(gl.path)}: final graph must retain at least one root")
        final_era_nodes = sorted(nid for nid in gl.nodes if gl.kind(nid) == "ERA")
        if final_era_nodes:
            errs.append(
                f"{os.path.basename(gl.path)}: final graph still contains ERA nodes {final_era_nodes[:6]}"
            )
        seen_final = set()
        for nid in gl.nodes:
            if nid in seen_final:
                continue
            q = [nid]
            seen_final.add(nid)
            comp = []
            while q:
                cur = q.pop(0)
                comp.append(cur)
                for nx in final_adj.get(cur, []):
                    if nx in seen_final:
                        continue
                    seen_final.add(nx)
                    q.append(nx)
            comp_roots = [n for n in comp if n in final_roots]
            if not comp_roots:
                errs.append(
                    f"{os.path.basename(gl.path)}: final graph has rootless component {sorted(comp)[:6]}"
                )

    for g in graphs:
        if g.suffix.startswith("state_no_highlight"):
            errs.append(f"{os.path.basename(g.path)}: hidden next interaction (state_no_highlight artifact)")

        # 0) every interaction step should mark the next reducible edge in red
        if g.step > 0 and not g.suffix.startswith("state_"):
            if not any("#cc0000" in attrs for _, _, attrs in g.edges):
                errs.append(f"{os.path.basename(g.path)}: missing highlighted next-interaction edge")

        # 1) no undeclared node refs
        for src, dst, _ in g.edges:
            if src not in g.nodes:
                errs.append(f"{os.path.basename(g.path)}: edge src '{src}' is undeclared")
            if dst not in g.nodes:
                errs.append(f"{os.path.basename(g.path)}: edge dst '{dst}' is undeclared")

        out_map: Dict[str, List[Tuple[str, str, str]]] = {}
        in_map: Dict[str, List[Tuple[str, str, str]]] = {}
        for e in g.edges:
            out_map.setdefault(e[0], []).append(e)
            in_map.setdefault(e[1], []).append(e)

        # 2a) Op arity / required ports
        for nid in g.nodes:
            if g.kind(nid) != "NODE":
                continue
            op = g.nodes[nid].split("\\n", 1)[0].strip().upper()
            if "\\n[]" in g.nodes[nid]:
                errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' has empty shape label []")
            sm = SHAPE_LABEL_RE.search(g.nodes[nid])
            if sm:
                toks = [t.strip() for t in sm.group(1).split(",") if t.strip()]
                if any(t == "0" for t in toks):
                    errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' has zero dimension in shape [{sm.group(1)}]")
            incoming = in_map.get(nid, [])
            in_labels = {edge_label(attrs) for _, _, attrs in incoming}
            in_count = len(incoming)
            out_labels = {edge_label(attrs) for _, _, attrs in out_map.get(nid, [])}

            if op in {"SUM", "RMAX"}:
                need = {"in", "axes"}
                miss = need - in_labels
                if miss:
                    errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' missing inputs {sorted(miss)}")
                if in_count != 2:
                    errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' has {in_count} inputs (expected 2)")
            elif op in VIEW_OPS:
                need = {"in", "shape"}
                miss = need - in_labels
                if miss:
                    errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' missing inputs {sorted(miss)}")
                if in_count != 2:
                    errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' has {in_count} inputs (expected 2)")
            elif op == "ASSIGN":
                errs.append(f"{os.path.basename(g.path)}: phase-1 graph must not contain ASSIGN node '{nid}'")
            elif op == "GRAD":
                need = {"y", "gy"}
                miss = need - in_labels
                if miss:
                    errs.append(f"{os.path.basename(g.path)}: GRAD node '{nid}' missing inputs {sorted(miss)}")
                if in_count != 2:
                    errs.append(f"{os.path.basename(g.path)}: GRAD node '{nid}' has {in_count} inputs (expected 2)")
                for src, _, attrs in incoming:
                    if edge_label(attrs) == "gy" and g.kind(src) == "TEN":
                        errs.append(f"{os.path.basename(g.path)}: GRAD node '{nid}' uses tensor '{src}' as visible gy seed")
            elif op in BINARY_OPS:
                if in_count != 2:
                    errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' has {in_count} inputs (expected 2)")
            elif op in UNARY_OPS or op in PASSTHRU_UNARY_OPS:
                if in_count != 1:
                    errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' has {in_count} inputs (expected 1)")

            # IC principal-port invariant: non-dup combinators are single-output.
            # In child->parent orientation this means at most one outgoing edge.
            out_count = len(out_map.get(nid, []))
            if out_count > 1:
                errs.append(
                    f"{os.path.basename(g.path)}: {op} node '{nid}' has {out_count} outputs (expected <=1); missing commute/split"
                )

        # 2) ERA is single-principal: no auxiliary fanout.
        for nid in g.nodes:
            if g.kind(nid) != "ERA":
                continue
            ins = in_map.get(nid, [])
            outs = out_map.get(nid, [])
            n_in = len(ins)
            n_out = len(outs)
            incident = n_in + n_out
            if incident == 0:
                errs.append(f"{os.path.basename(g.path)}: ERA node '{nid}' is isolated")
                continue
            labels = []
            payload_edges = 0
            parent_edges = 0
            for _, _, attrs in ins + outs:
                lbl = edge_label(attrs)
                if lbl:
                    labels.append(lbl)
                if lbl in ("p", "dp0", "dp1"):
                    payload_edges += 1
                else:
                    parent_edges += 1
            # Detached active ERA: payload -> ERA
            if incident == 1:
                # Single-edge inline ERA under a parent op is also valid once
                # the forward net stays fully lazy in phase 1.
                continue
            # Inline active ERA under a parent op: payload -> ERA -> parent
            if incident == 2 and payload_edges == 1 and parent_edges == 1:
                continue
            errs.append(
                f"{os.path.basename(g.path)}: ERA node '{nid}' has {incident} incident edges "
                f"(labels={labels}); invalid ERA form"
            )

        # 3) No plain node/tensor fanout directly to ERA and another target.
        for nid in g.nodes:
            k = g.kind(nid)
            if k in ("ERA", "DUP", "TEN", "NUM", "ANY", "CTR"):
                continue
            if RAW_HEAP_NODE_RE.match(nid):
                continue
            outs = out_map.get(nid, [])
            if not outs:
                continue
            to_era = any(g.kind(dst) == "ERA" for _, dst, _ in outs)
            to_non_era = any(g.kind(dst) != "ERA" for _, dst, _ in outs)
            if to_era and to_non_era:
                errs.append(f"{os.path.basename(g.path)}: node '{nid}' fans out to ERA and non-ERA targets")

        # 4) DUP used for ERA must still show another consumer.
        for nid in g.nodes:
            if g.kind(nid) != "DUP":
                continue
            outs = out_map.get(nid, [])
            if not outs:
                errs.append(f"{os.path.basename(g.path)}: DUP '{nid}' has no outgoing edges")
                continue
            ports = {edge_dp_port(attrs) for _, _, attrs in outs}
            port_counts = {"dp0": 0, "dp1": 0}
            port_self_counts = {"dp0": 0, "dp1": 0}
            for src, dst, attrs in outs:
                p = edge_dp_port(attrs)
                if p in port_counts:
                    # Self-loop DUP->DUP edges are placeholders for unresolved alias ports.
                    # They satisfy "port exists" but should not count as extra consumers.
                    if src == nid and dst == nid:
                        port_self_counts[p] += 1
                    else:
                        port_counts[p] += 1
            has_dp0 = ("dp0" in ports) or (port_self_counts["dp0"] > 0)
            has_dp1 = ("dp1" in ports) or (port_self_counts["dp1"] > 0)
            if port_counts["dp0"] > 1:
                errs.append(f"{os.path.basename(g.path)}: DUP '{nid}' has {port_counts['dp0']} dp0 consumers (expected <=1)")
            if port_counts["dp1"] > 1:
                errs.append(f"{os.path.basename(g.path)}: DUP '{nid}' has {port_counts['dp1']} dp1 consumers (expected <=1)")
            if not has_dp0:
                errs.append(f"{os.path.basename(g.path)}: DUP '{nid}' is missing dp0 output")
            if not has_dp1:
                errs.append(f"{os.path.basename(g.path)}: DUP '{nid}' is missing dp1 output")
        # 4b) Isolated tensors indicate heap-dump artifacts or broken links.
        for nid in g.nodes:
            if g.kind(nid) != "TEN":
                continue
            if len(in_map.get(nid, [])) == 0 and len(out_map.get(nid, [])) == 0:
                errs.append(f"{os.path.basename(g.path)}: tensor '{nid}' is isolated")

        # 4a) Disconnected non-ERA components with operation nodes are suspicious.
        # Skip strict disconnectedness on explicit ERA-interaction steps:
        # these snapshots may temporarily isolate pending-erasure subnets.
        if not g.suffix.startswith("interact_era_on_"):
            adj: Dict[str, List[str]] = {}
            for nid in g.nodes:
                if RAW_HEAP_NODE_RE.match(nid):
                    continue
                adj[nid] = []
            for src, dst, _ in g.edges:
                if src not in adj or dst not in adj:
                    continue
                adj[src].append(dst)
                adj[dst].append(src)
            comps: List[List[str]] = []
            seen_nodes = set()
            for nid in adj:
                if nid in seen_nodes:
                    continue
                q = [nid]
                seen_nodes.add(nid)
                comp = []
                while q:
                    cur = q.pop(0)
                    comp.append(cur)
                    for nx in adj[cur]:
                        if nx in seen_nodes:
                            continue
                        seen_nodes.add(nx)
                        q.append(nx)
                comps.append(comp)
            if comps:
                main_comp = max(comps, key=len)
                main_set = set(main_comp)
                for comp in comps:
                    cset = set(comp)
                    if cset == main_set:
                        continue
                    has_erase = any(g.kind(n) == "ERA" for n in comp)
                    has_op = any(g.kind(n) == "NODE" for n in comp)
                    has_assign = any(node_head(g, n) == "ASSIGN" for n in comp)
                    if has_op and not has_erase and not has_assign:
                        errs.append(
                            f"{os.path.basename(g.path)}: disconnected non-ERA component with ops: {sorted(comp)[:4]}"
                        )

    # 4c) Highlighted next interaction metadata must match the interaction
    # that actually produced the following step.
    for a, b in zip(graphs, graphs[1:]):
        b_prev = prev_interaction_name(b)
        if not a.next_interaction:
            errs.append(f"{os.path.basename(a.path)}: missing NEXT_INTERACTION metadata")
            continue
        if not b_prev:
            errs.append(f"{os.path.basename(b.path)}: missing PREV_INTERACTION metadata")
            continue
        if not interactions_match(a.next_interaction, b_prev):
            errs.append(
                f"{os.path.basename(a.path)}: highlighted next interaction '{a.next_interaction}' "
                f"does not match following step '{b_prev}'"
            )

    # 5) no identical consecutive steps
    for a, b in zip(graphs, graphs[1:]):
        if a.fingerprint() == b.fingerprint():
            errs.append(
                f"{os.path.basename(a.path)} and {os.path.basename(b.path)} are identical consecutive steps"
            )

    # 6) Tensor disappearance must be justified by explicit ERA interaction
    # in that same step graph (directly or through DUP path to ERA).
    for a, b in zip(graphs, graphs[1:]):
        # ERA interaction steps are allowed to remove tensors as part of
        # principal-port erasure even if the source-side path is not rendered.
        if b.suffix.startswith("interact_era_on_"):
            continue
        out_map: Dict[str, List[Tuple[str, str, str]]] = {}
        for e in a.edges:
            out_map.setdefault(e[0], []).append(e)
        ta = {nid for nid in a.nodes if a.kind(nid) == "TEN"}
        tb = {nid for nid in b.nodes if b.kind(nid) == "TEN"}
        vanished = ta - tb
        if not vanished:
            continue
        for tid in sorted(vanished):
            if not has_erase_path_from_tensor(a, tid, out_map):
                errs.append(
                    f"{os.path.basename(a.path)} -> {os.path.basename(b.path)}: tensor '{tid}' vanished without explicit ERA interaction"
                )
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate TinyHVM phase-1 step DOT graphs.")
    ap.add_argument("step_dir", nargs="?", default="thvm_steps")
    args = ap.parse_args()

    graphs = collect_graphs(args.step_dir)
    if not graphs:
        print(f"no step dot files found in {args.step_dir}", file=sys.stderr)
        return 2

    errs = check_graphs(graphs)
    if errs:
        print(f"FAIL: {len(errs)} graph invariant violations")
        for e in errs:
            print(f"  - {e}")
        return 1

    print(f"PASS: checked {len(graphs)} step graphs in {args.step_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
