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
    assign_steps = []
    fuse_nodes_by_file = {}
    edge_attrs_by_file = {}
    has_fuse_null = []
    has_nested_mul_kernel = []

    for path in step_files:
        m = STEP_RE.match(path.stem)
        if not m:
            continue
        idx = int(m.group(1))
        name = path.name
        text = path.read_text(encoding="utf-8", errors="replace")
        fuse_nodes = set()
        edge_attrs = []
        for line in text.splitlines():
            nm = NODE_RE.match(line)
            if nm:
                nid, attrs = nm.groups()
                lm = LABEL_RE.search(attrs or "")
                lbl = lm.group(1) if lm else ""
                head = lbl.split("\\n", 1)[0]
                if head == "FUSE":
                    fuse_nodes.add(nid)
                if "FUSE\\nNULL" in lbl:
                    has_fuse_null.append(name)
                if "KERNEL\\nMUL" in lbl:
                    has_nested_mul_kernel.append(name)
                continue
            em = EDGE_RE.match(line)
            if em:
                edge_attrs.append(em.groups())
        fuse_nodes_by_file[name] = fuse_nodes
        edge_attrs_by_file[name] = edge_attrs
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
    if has_nested_mul_kernel:
        return fail(f"{step_dir} still shows nested KERNEL MUL nodes: {has_nested_mul_kernel[:3]}")

    first_kernel = min(kernel_label_steps)
    if assign_steps and first_kernel >= min(assign_steps):
        return fail(
            f"first visible KERNEL step ({first_kernel}) does not precede first ASSIGN step ({min(assign_steps)})"
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
        f"PASS: visible KERNEL step at {first_kernel}"
        + (f" before ASSIGN step {min(assign_steps)}" if assign_steps else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
