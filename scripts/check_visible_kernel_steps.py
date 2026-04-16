#!/usr/bin/env python3
import re
import sys
from pathlib import Path


STEP_RE = re.compile(r"step_(\d+)_")
NODE_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*\[(.*)\]\s*;\s*$')
EDGE_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*(?:\[(.*)\])?\s*;\s*$')
LABEL_RE = re.compile(r'label="([^"]*)"')


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_visible_kernel_steps.py <step-dir>")
        return 2

    step_dir = Path(sys.argv[1])
    if not step_dir.is_dir():
        return fail(f"{step_dir} is not a directory")

    step_files = sorted(step_dir.glob("step_*.dot"))
    if not step_files:
        return fail(f"no step graphs found in {step_dir}")

    kernel_steps = []
    visible_kernel_steps = []
    kernel_label_steps = []
    monolithic_kernel_steps = []
    assign_steps = []
    fuse_nodes_by_file = {}
    edge_attrs_by_file = {}
    node_heads_by_file = {}
    has_fuse_null = []

    for path in step_files:
        m = STEP_RE.match(path.stem)
        if not m:
            continue
        idx = int(m.group(1))
        name = path.name
        text = path.read_text(encoding="utf-8", errors="replace")
        fuse_nodes = set()
        edge_attrs = []
        node_heads = []
        for line in text.splitlines():
            nm = NODE_RE.match(line)
            if nm:
                nid, attrs = nm.groups()
                lm = LABEL_RE.search(attrs or "")
                lbl = lm.group(1) if lm else ""
                head = lbl.split("\\n", 1)[0]
                node_heads.append(head)
                if head == "FUSE":
                    fuse_nodes.add(nid)
                if "FUSE\\nNULL" in lbl:
                    has_fuse_null.append(name)
                if head == "KERNEL":
                    parts = lbl.split("\\n")
                    if len(parts) >= 2 and "+" in parts[1]:
                        monolithic_kernel_steps.append(idx)
                continue
            em = EDGE_RE.match(line)
            if em:
                edge_attrs.append(em.groups())
        fuse_nodes_by_file[name] = fuse_nodes
        edge_attrs_by_file[name] = edge_attrs
        node_heads_by_file[name] = node_heads
        if "_KERNEL_" in name:
            kernel_steps.append(idx)
        if "label=\"KERNEL" in text:
            kernel_label_steps.append(idx)
            if "_KERNEL_" in name:
                visible_kernel_steps.append(idx)
        if "_ASSIGN_" in name:
            assign_steps.append(idx)

    if not kernel_label_steps:
        return fail(f"{step_dir} has no visible KERNEL node labels")
    if kernel_steps and not visible_kernel_steps:
        return fail(f"{step_dir} has KERNEL step filenames but no visible KERNEL node labels")
    if has_fuse_null:
        return fail(f"{step_dir} still renders FUSE NULL payload labels: {has_fuse_null[:3]}")

    if not monolithic_kernel_steps:
        return fail(f"{step_dir} never exposes a monolithic multi-op KERNEL node")
    first_monolithic = min(monolithic_kernel_steps)
    first_kernel = min(kernel_label_steps)
    if first_kernel >= first_monolithic:
        return fail(
            f"{step_dir} still collapses directly to a monolithic KERNEL without visible growing KERNEL buildup"
        )
    last_graph_text = step_files[-1].read_text(encoding="utf-8", errors="replace")
    if 'label="KERNEL' not in last_graph_text or '+' not in last_graph_text:
        return fail(f"{step_dir} does not end in a monolithic KERNEL state")

    raw_compute_heads = {
        "ADD", "SUB", "MUL", "DIV", "MAX", "CMP",
        "NEG", "RELU", "EXP", "LOG", "SQRT", "CAST",
        "SUM", "RMAX", "RESHAPE", "PERMUTE", "EXPAND",
        "SHRINK", "PAD", "MM",
    }
    raw_compute_after_kernel = []
    fuse_after_kernel = []
    for path in step_files:
        m = STEP_RE.match(path.stem)
        if not m:
            continue
        idx = int(m.group(1))
        if idx < first_monolithic:
            continue
        heads = node_heads_by_file.get(path.name, [])
        bad = sorted({head for head in heads if head in raw_compute_heads})
        if bad:
            raw_compute_after_kernel.append(f"{path.name}: {','.join(bad)}")
        if fuse_nodes_by_file.get(path.name):
            fuse_after_kernel.append(path.name)

    if raw_compute_after_kernel:
        return fail(f"{step_dir} still shows raw compute after first monolithic KERNEL step: {raw_compute_after_kernel[:3]}")
    if fuse_after_kernel:
        return fail(f"{step_dir} still shows FUSE nodes after first monolithic KERNEL step: {fuse_after_kernel[:3]}")

    if assign_steps and first_monolithic >= min(assign_steps):
        return fail(
            f"first monolithic KERNEL step ({first_monolithic}) does not precede first ASSIGN step ({min(assign_steps)})"
        )

    for name, edges in edge_attrs_by_file.items():
        fuse_nodes = fuse_nodes_by_file.get(name, set())
        if not fuse_nodes:
            continue
        for src, dst, attrs in edges:
            if dst not in fuse_nodes:
                continue
            lm = LABEL_RE.search(attrs or "")
            label = lm.group(1) if lm else ""
            if not label.startswith("in"):
                return fail(f"{name}: incoming edge to {dst} missing 'in' port label")

    print(
        f"PASS: local KERNEL buildup reaches monolithic KERNEL step at {first_monolithic}"
        + (f" before ASSIGN step {min(assign_steps)}" if assign_steps else "")
        + f"; first visible KERNEL step is {first_kernel}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
