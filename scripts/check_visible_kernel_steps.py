#!/usr/bin/env python3
import re
import sys
from pathlib import Path


STEP_RE = re.compile(r"step_(\d+)_")


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
    assign_steps = []

    for path in step_files:
        m = STEP_RE.match(path.stem)
        if not m:
            continue
        idx = int(m.group(1))
        name = path.name
        text = path.read_text(encoding="utf-8", errors="replace")
        if "_KERNEL_" in name:
            kernel_steps.append(idx)
            if "label=\"KERNEL" in text:
                visible_kernel_steps.append(idx)
        if "_ASSIGN_" in name:
            assign_steps.append(idx)

    if not kernel_steps:
        return fail(f"{step_dir} has no KERNEL step filenames")
    if not visible_kernel_steps:
        return fail(f"{step_dir} has KERNEL step filenames but no visible KERNEL node labels")

    first_kernel = min(visible_kernel_steps)
    if assign_steps and first_kernel >= min(assign_steps):
        return fail(
            f"first visible KERNEL step ({first_kernel}) does not precede first ASSIGN step ({min(assign_steps)})"
        )

    print(
        f"PASS: visible KERNEL step at {first_kernel}"
        + (f" before ASSIGN step {min(assign_steps)}" if assign_steps else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
