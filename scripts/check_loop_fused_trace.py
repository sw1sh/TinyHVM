#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path


STEP_RE = re.compile(r"^step_(\d+)_(.+)\.dot$")
ITER_RE = re.compile(r"n(\d+)_")
FUSE_TEN_RE = re.compile(r"(?:^|_)FUSE_h\d+_TEN_h\d+$")
ERA_SEQ_RE = re.compile(r"^ERA_h\d+_SEQ_h\d+$")


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


def interaction_name(step_name: str) -> str:
    return step_name.split("_h", 1)[0]

def local_assign_admin_step(step_name: str) -> bool:
    return "_ALO_" in step_name or "_DP0_" in step_name or "_DP1_" in step_name


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

    if args.diag_log:
        diag_path = Path(args.diag_log)
        if not diag_path.is_file():
            return fail(f"diag log {diag_path} does not exist")

    bad_fuse_ten = []
    for i, (_, name, _) in enumerate(steps):
        if not FUSE_TEN_RE.search(name):
            continue
        next_name = steps[i + 1][1] if i + 1 < len(steps) else "state_final"
        if next_name.startswith("SWEEP_") or next_name.startswith("state_final"):
            continue
        bad_fuse_ten.append(name)
    if bad_fuse_ten:
        return fail(f"{step_dir} still contains non-handoff FUSE->TEN steps: {bad_fuse_ten[:3]}")

    m_iter = ITER_RE.search(step_dir.name)
    train_steps = int(m_iter.group(1)) if m_iter else None
    kernel_steps = [idx for idx, name, _ in steps if interaction_name(name) == "KERNEL"]
    visible_kernel_steps = [
        idx for idx, _, path in steps
        if 'label="KERNEL' in path.read_text(encoding="utf-8", errors="replace")
    ]
    assign_steps = [
        idx for idx, name, _ in steps
        if interaction_name(name) == "ASSIGN" and not local_assign_admin_step(name)
    ]
    if assign_steps:
        return fail(
            f"{step_dir} still contains ASSIGN interaction steps during local fused tracing: {assign_steps[:4]}"
        )
    if train_steps is not None and train_steps >= 2 and not (kernel_steps or visible_kernel_steps):
        return fail(f"{step_dir} never exposes a visible KERNEL node in local fused tracing")

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

    print(
        f"PASS: fused loop trace stays in local coarse phase in {step_dir}"
        + (f" (diag log: {args.diag_log})" if args.diag_log else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
