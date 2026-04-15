#!/usr/bin/env python3
import argparse
import re
import sys
from collections import Counter
from pathlib import Path


STEP_RE = re.compile(r"^step_(\d+)_(.+)\.dot$")
FUSE_TEN_RE = re.compile(r"(?:^|_)FUSE_h\d+_TEN_h\d+$")
ERA_SEQ_RE = re.compile(r"^ERA_h\d+_SEQ_h\d+$")
DISPATCH_RE = re.compile(r"KERNEL_DISPATCH kid=\d+ sig=(0x[0-9a-fA-F]+)")
STALE_RE = re.compile(r"KERNEL_CACHE_STALE kid=\d+ sig=(0x[0-9a-fA-F]+)")
ASSIGN_RE = re.compile(r"^ASSIGN:")


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def parse_steps(step_dir: Path):
    steps = []
    for path in sorted(step_dir.glob("step_*.dot")):
        m = STEP_RE.match(path.name)
        if not m:
            continue
        steps.append((int(m.group(1)), m.group(2), path))
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description="Check fused loop step traces.")
    ap.add_argument("step_dir")
    ap.add_argument("--diag-log", dest="diag_log")
    args = ap.parse_args()

    step_dir = Path(args.step_dir)
    if not step_dir.is_dir():
        return fail(f"{step_dir} is not a directory")

    steps = parse_steps(step_dir)
    if not steps:
        return fail(f"no step graphs found in {step_dir}")

    fuse_ten = [name for _, name, _ in steps if FUSE_TEN_RE.search(name)]
    if fuse_ten:
        return fail(f"{step_dir} still contains administrative FUSE->TEN steps: {fuse_ten[:3]}")

    kernel_steps = [idx for idx, name, _ in steps if "_KERNEL_" in f"_{name}_"]
    assign_steps = [idx for idx, name, _ in steps if "_ASSIGN_" in f"_{name}_"]
    if kernel_steps and assign_steps and not any(k < a for k in kernel_steps for a in assign_steps):
        return fail(
            f"{step_dir} has KERNEL steps {kernel_steps[:4]} but none precede an ASSIGN step {assign_steps[:4]}"
        )

    trailing_cleanup = []
    for idx, name, _ in reversed(steps):
        if name.startswith("state_final"):
            continue
        if ERA_SEQ_RE.match(name):
            trailing_cleanup.append((idx, name))
            continue
        break
    if trailing_cleanup:
        trailing_cleanup.reverse()
        return fail(f"{step_dir} still ends with ERA/SEQ cleanup tail: {trailing_cleanup[:3]}")

    if args.diag_log:
        diag_path = Path(args.diag_log)
        if not diag_path.is_file():
            return fail(f"diag log {diag_path} does not exist")
        text = diag_path.read_text(encoding="utf-8", errors="replace")
        dispatch_counts = Counter(DISPATCH_RE.findall(text))
        repeated = {sig: n for sig, n in dispatch_counts.items() if n >= 2}
        if not repeated:
            return fail(f"{diag_path} does not show any repeated KERNEL_DISPATCH signature")
        assign_seen = False
        dispatch_before_assign = set()
        dispatch_after_assign = set()
        for line in text.splitlines():
            if ASSIGN_RE.search(line):
                assign_seen = True
                continue
            m = DISPATCH_RE.search(line)
            if not m:
                continue
            sig = m.group(1)
            if not assign_seen:
                dispatch_before_assign.add(sig)
            else:
                dispatch_after_assign.add(sig)
        if not (dispatch_before_assign & dispatch_after_assign):
            return fail(f"{diag_path} has repeated dispatch signatures but none occur on both sides of an ASSIGN")

    print(
        f"PASS: fused loop trace checks OK in {step_dir}"
        + (f" with diag {args.diag_log}" if args.diag_log else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
