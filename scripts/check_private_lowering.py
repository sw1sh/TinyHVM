#!/usr/bin/env python3
import re
import sys
from pathlib import Path


LOWER_BUILT_RE = re.compile(
    r"LOWER_KERNEL_BUILT kid=\d+ cache_sig=(0x[0-9a-fA-F]+) "
    r"lower_sig=(0x[0-9a-fA-F]+) lower_heap=(\d+) ops=(\d+) rewrites=(\d+)"
)
LOWER_FALLBACK_RE = re.compile(r"LOWER_KERNEL_FALLBACK ")

LOWER_TOKENS = (
    "LSINK",
    "LGID",
    "LCONST_",
    "LRANGE",
    "LENDRANGE",
    "LLOAD",
    "LSTORE",
    "LALU",
    "LACC",
    "LINDEX",
    "LMOD",
    "LDIV",
    "LMASK",
)


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: check_private_lowering.py <diag-log> [step-dir]")
        return 2

    log_path = Path(sys.argv[1])
    if not log_path.is_file():
        return fail(f"{log_path} does not exist")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    built = LOWER_BUILT_RE.findall(text)
    if not built:
        return fail(f"{log_path} has no LOWER_KERNEL_BUILT lines")
    if LOWER_FALLBACK_RE.search(text):
        return fail(f"{log_path} still reports LOWER_KERNEL_FALLBACK")

    for cache_sig, lower_sig, lower_heap, ops, rewrites in built:
        if int(ops) <= 0:
            return fail(f"{log_path} reported non-positive lowered op count for {cache_sig}")
        if int(lower_heap) <= 1:
            return fail(f"{log_path} reported empty lower heap for {cache_sig}")
        if lower_sig == "0x0000000000000000":
            return fail(f"{log_path} reported zero lower signature for {cache_sig}")

    if len(sys.argv) == 3:
        step_dir = Path(sys.argv[2])
        if not step_dir.is_dir():
            return fail(f"{step_dir} is not a directory")
        step_files = sorted(step_dir.glob("step_*.dot"))
        if not step_files:
            return fail(f"{step_dir} has no step graphs")
        for path in step_files:
            dot = path.read_text(encoding="utf-8", errors="replace")
            for token in LOWER_TOKENS:
                if token in dot:
                    return fail(f"{path.name} leaked private lowering token {token} into public graph")

    print(
        "PASS: private lowering built "
        f"{len(built)} kernel(s) without fallback"
        + (f" and stayed out of public step graphs" if len(sys.argv) == 3 else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
