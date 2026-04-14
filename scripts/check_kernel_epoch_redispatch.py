#!/usr/bin/env python3
import re
import sys
from collections import Counter
from pathlib import Path


DISPATCH_RE = re.compile(r"KERNEL_DISPATCH kid=\d+ sig=(0x[0-9a-fA-F]+)")
HIT_RE = re.compile(r"KERNEL_CACHE_HIT kid=\d+ sig=(0x[0-9a-fA-F]+)")
STALE_RE = re.compile(r"KERNEL_CACHE_STALE kid=\d+ sig=(0x[0-9a-fA-F]+)")
ASSIGN_RE = re.compile(r"^ASSIGN:")


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_kernel_epoch_redispatch.py <diag-log>")
        return 2

    log_path = Path(sys.argv[1])
    if not log_path.is_file():
        return fail(f"{log_path} does not exist")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    dispatch = Counter(DISPATCH_RE.findall(text))
    hits = Counter(HIT_RE.findall(text))
    stale = Counter(STALE_RE.findall(text))

    if not dispatch:
        return fail(f"{log_path} has no KERNEL_DISPATCH lines")

    candidates = [sig for sig, count in dispatch.items() if count >= 2]
    if not candidates:
        return fail(f"{log_path} has no signature dispatched more than once")

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

    for sig in candidates:
        if sig in dispatch_before_assign and sig in dispatch_after_assign:
            print(
                f"PASS: redispatch observed for {sig} "
                f"(dispatch={dispatch[sig]} hit={hits[sig]} stale={stale[sig]})"
            )
            return 0

    return fail(f"{log_path} has repeated dispatches {candidates[:3]} but none straddle an ASSIGN")


if __name__ == "__main__":
    raise SystemExit(main())
