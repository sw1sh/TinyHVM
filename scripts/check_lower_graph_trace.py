#!/usr/bin/env python3
import sys
from pathlib import Path


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_lower_graph_trace.py <lower-dir>")
        return 2

    lower_dir = Path(sys.argv[1])
    if not lower_dir.is_dir():
        return fail(f"{lower_dir} is not a directory")

    dot_files = sorted(lower_dir.glob("step_*.dot"))
    txt_files = sorted(lower_dir.glob("step_*.txt"))
    if not dot_files:
        return fail(f"{lower_dir} has no lower DOT dumps")
    if not txt_files:
        return fail(f"{lower_dir} has no lower manifest dumps")

    fused_manifest = None
    fused_manifest_text = ""
    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if ("fused_ops:" in text and
                "MUL" in text and
                "ADD" in text and
                "dispatch_mode=" in text and
                "output_slot=" in text and
                "grid=[" in text):
            fused_manifest = path
            fused_manifest_text = text
            break
    if fused_manifest is None:
        return fail(f"{lower_dir} has no manifest showing fused MUL+ADD kernel metadata")

    fused_dot = None
    for path in dot_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if ("LowerCtx IC" in text and
                "Fused Ops" in text and
                "Memory Plan" in text and
                "LLOAD" in text and
                "LSTORE" in text and
                "KOP_ALU" in text and
                "MUL" in text and
                "ADD" in text):
            fused_dot = path
            break
    if fused_dot is None:
        return fail(f"{lower_dir} has no DOT dump showing fused MUL+ADD lowering structure")

    png_files = sorted(lower_dir.glob("step_*.png"))
    if not png_files:
        return fail(f"{lower_dir} has no rendered PNG dumps")

    if "backend=mtl" in fused_manifest_text:
        msl_path = fused_manifest.with_suffix(".msl")
        if not msl_path.is_file():
            return fail(f"{lower_dir} is missing Metal source dump for {fused_manifest.name}")
        msl = msl_path.read_text(encoding="utf-8", errors="replace")
        if "kernel void K" not in msl or "buf0" not in msl:
            return fail(f"{msl_path.name} does not look like rendered Metal source")

    print(
        f"PASS: lower trace captured fused MUL+ADD kernel in {fused_dot.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
