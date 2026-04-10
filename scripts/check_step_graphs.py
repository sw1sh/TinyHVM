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

BINARY_OPS = {
    "ADD", "SUB", "MUL", "DIV", "MAX", "CMP", "MM",
    "EQL", "AND", "OR", "MAT"
}
UNARY_OPS = {"NEG", "RELU", "EXP", "LOG", "SQRT"}
VIEW_OPS = {"RESHAPE", "PERMUTE", "EXPAND", "SHRINK", "PAD"}


@dataclass
class DotGraph:
    path: str
    step: int
    suffix: str
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


def check_graphs(graphs: List[DotGraph]) -> List[str]:
    errs: List[str] = []
    for g in graphs:
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
            incoming = in_map.get(nid, [])
            in_labels = {edge_label(attrs) for _, _, attrs in incoming}
            in_count = len(incoming)
            out_labels = {edge_label(attrs) for _, _, attrs in out_map.get(nid, [])}

            if op in {"SUM", "RMAX"}:
                need = {"in", "axes"}
                miss = need - in_labels
                if miss:
                    errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' missing inputs {sorted(miss)}")
            elif op in VIEW_OPS:
                need = {"in", "shape"}
                miss = need - in_labels
                if miss:
                    errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' missing inputs {sorted(miss)}")
            elif op == "ASSIGN":
                need = {"tgt", "src"}
                miss = need - in_labels
                if miss:
                    errs.append(f"{os.path.basename(g.path)}: ASSIGN node '{nid}' missing inputs {sorted(miss)}")
            elif op == "GRAD":
                if "y" not in in_labels:
                    errs.append(f"{os.path.basename(g.path)}: GRAD node '{nid}' missing 'y' input")
            elif op in BINARY_OPS:
                if in_count < 2:
                    errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' has {in_count} inputs (expected >=2)")
            elif op in UNARY_OPS:
                if in_count < 1:
                    errs.append(f"{os.path.basename(g.path)}: {op} node '{nid}' has {in_count} inputs (expected >=1)")

        # 2) ERA is single-principal in visualization: one incoming, no outgoing
        for nid in g.nodes:
            if g.kind(nid) != "ERA":
                continue
            n_in = len(in_map.get(nid, []))
            n_out = len(out_map.get(nid, []))
            if n_in != 1:
                errs.append(f"{os.path.basename(g.path)}: ERA node '{nid}' has {n_in} incoming edges (expected 1)")
            if n_out != 0:
                errs.append(f"{os.path.basename(g.path)}: ERA node '{nid}' has {n_out} outgoing edges (expected 0)")

        # 3) No plain node/tensor fanout directly to ERA and another target.
        for nid in g.nodes:
            k = g.kind(nid)
            if k in ("ERA", "DUP"):
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
            has_era = any(g.kind(dst) == "ERA" for _, dst, _ in outs)
            has_non_era = any(g.kind(dst) != "ERA" for _, dst, _ in outs)
            if has_era and not has_non_era:
                errs.append(f"{os.path.basename(g.path)}: DUP '{nid}' has only ERA output")

        # 4a) Disconnected non-ERA components with operation nodes are suspicious.
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
                has_era = any(g.kind(n) == "ERA" for n in comp)
                has_op = any(g.kind(n) == "NODE" for n in comp)
                if has_op and not has_era:
                    errs.append(
                        f"{os.path.basename(g.path)}: disconnected non-ERA component with ops: {sorted(comp)[:4]}"
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
