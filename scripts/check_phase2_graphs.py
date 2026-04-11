#!/usr/bin/env python3
import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

NODE_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*\[(.*)\]\s*;\s*$')
LABEL_RE = re.compile(r'label="([^"]*)"')
TENSOR_HEAD_RE = re.compile(r'^t\d+$')
NUM_HEAD_RE = re.compile(r'^-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$', re.IGNORECASE)
KERNEL_HEAD_RE = re.compile(r'^K\d+: ')

FORBIDDEN_HEADS = {
    "RESHAPE", "PERMUTE", "EXPAND", "SHRINK", "PAD",
    "GRAD", "ASSIGN", "ERA",
}


@dataclass
class DotGraph:
    path: str
    nodes: Dict[str, str] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)


def parse_dot(path: str) -> DotGraph:
    g = DotGraph(path=path)
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
            if "->" in line and ";" in line:
                parts = line.strip().split("->", 1)
                if len(parts) == 2:
                    src = parts[0].strip()
                    dst = parts[1].split("[", 1)[0].split(";", 1)[0].strip()
                    g.edges.append((src, dst))
    return g


def head(label: str) -> str:
    return label.split("\\n", 1)[0].strip()


def is_allowed_head(h: str) -> bool:
    if h in {"LOG_PRINT", "CTR"}:
        return True
    if KERNEL_HEAD_RE.match(h):
        return True
    if TENSOR_HEAD_RE.match(h):
        return True
    if NUM_HEAD_RE.match(h):
        return True
    return False


def is_kernel_head(h: str) -> bool:
    return bool(KERNEL_HEAD_RE.match(h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_dir")
    ap.add_argument("--expect-log", type=int, default=None)
    ap.add_argument("--expect-kernels", type=int, default=None)
    ap.add_argument("--min-kernels", type=int, default=1)
    ap.add_argument("--allow-ctr", action="store_true")
    ap.add_argument("--expect-ctr", type=int, default=None)
    args = ap.parse_args()

    graph_dir = args.graph_dir
    pre = os.path.join(graph_dir, "thvm_0_pre_reduce.dot")
    post_reduce = os.path.join(graph_dir, "thvm_1_post_reduce.dot")
    post_sched = os.path.join(graph_dir, "thvm_2_post_sched.dot")

    missing = [p for p in (pre, post_reduce, post_sched) if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"missing phase2 graph: {p}", file=sys.stderr)
        return 1

    g_reduce = parse_dot(post_reduce)
    g_sched = parse_dot(post_sched)

    errs: List[str] = []

    if len(g_reduce.nodes) == 0:
        if len(g_sched.nodes) != 0:
            errs.append(
                f"{os.path.basename(post_sched)}: post_reduce is empty, expected empty post_sched "
                f"but found {len(g_sched.nodes)} nodes"
            )
    elif len(g_sched.nodes) >= len(g_reduce.nodes):
        errs.append(
            f"{os.path.basename(post_sched)}: post_sched has {len(g_sched.nodes)} nodes, "
            f"expected fewer than post_reduce's {len(g_reduce.nodes)}"
        )

    kernel_count = 0
    log_count = 0
    ctr_count = 0

    for nid, label in g_sched.nodes.items():
        h = head(label)
        if h in FORBIDDEN_HEADS:
            errs.append(f"{os.path.basename(post_sched)}: forbidden post_sched node '{h}' at {nid}")
            continue
        if h == "LOG_PRINT":
            log_count += 1
        elif h == "CTR":
            ctr_count += 1
            if not args.allow_ctr:
                errs.append(f"{os.path.basename(post_sched)}: unexpected CTR node at {nid}")
        elif KERNEL_HEAD_RE.match(h):
            kernel_count += 1
        elif not is_allowed_head(h):
            errs.append(f"{os.path.basename(post_sched)}: unexpected node '{h}' at {nid}")

    for src, dst in g_sched.edges:
        sh = head(g_sched.nodes.get(src, ""))
        dh = head(g_sched.nodes.get(dst, ""))
        if is_kernel_head(sh) and is_kernel_head(dh):
            errs.append(
                f"{os.path.basename(post_sched)}: direct kernel-to-kernel edge {src}->{dst} is forbidden"
            )

    if args.expect_kernels is not None and kernel_count != args.expect_kernels:
        errs.append(
            f"{os.path.basename(post_sched)}: found {kernel_count} kernels, "
            f"expected {args.expect_kernels}"
        )
    elif kernel_count < args.min_kernels:
        errs.append(
            f"{os.path.basename(post_sched)}: found {kernel_count} kernels, "
            f"expected at least {args.min_kernels}"
        )

    if args.expect_log is not None and log_count != args.expect_log:
        errs.append(
            f"{os.path.basename(post_sched)}: found {log_count} LOG_PRINT nodes, "
            f"expected {args.expect_log}"
        )

    if args.expect_ctr is not None and ctr_count != args.expect_ctr:
        errs.append(
            f"{os.path.basename(post_sched)}: found {ctr_count} CTR nodes, "
            f"expected {args.expect_ctr}"
        )

    if errs:
        for err in errs:
            print(err, file=sys.stderr)
        return 1

    print(
        f"PASS: {post_sched} kernels={kernel_count} log={log_count} ctr={ctr_count} "
        f"nodes={len(g_sched.nodes)} < post_reduce={len(g_reduce.nodes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
