#!/usr/bin/env python3
import re
import sys
from pathlib import Path


KID_SUFFIX_RE = re.compile(r"_kid\d+$")


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def shared_artifact_path(manifest_path: Path, suffix: str) -> Path:
    stem = KID_SUFFIX_RE.sub("", manifest_path.stem)
    return manifest_path.with_name(f"{stem}{suffix}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_lower_graph_trace.py <lower-dir>")
        return 2

    lower_dir = Path(sys.argv[1])
    if not lower_dir.is_dir():
        return fail(f"{lower_dir} is not a directory")

    dot_files = sorted(lower_dir.glob("step_*.dot"))
    txt_files = sorted(lower_dir.glob("step_*.txt"))
    png_files = sorted(lower_dir.glob("step_*.png"))
    if not dot_files:
        return fail(f"{lower_dir} has no lower DOT dumps")
    if not txt_files:
        return fail(f"{lower_dir} has no lower manifest dumps")
    if len(txt_files) != 1:
        return fail(f"{lower_dir} should keep exactly one monolithic manifest, found {len(txt_files)}")
    if len(dot_files) != 1:
        return fail(f"{lower_dir} should keep exactly one monolithic lower DOT graph, found {len(dot_files)}")
    if len(png_files) != 1:
        return fail(f"{lower_dir} should keep exactly one monolithic lower PNG graph, found {len(png_files)}")
    shared_dot_text = dot_files[0].read_text(encoding="utf-8", errors="replace")
    if "MUL" not in shared_dot_text or "ADD" not in shared_dot_text:
        return fail(f"{dot_files[0].name} should include both MUL and ADD in the monolithic lowering graph")

    manifest = txt_files[0]
    manifest_text = manifest.read_text(encoding="utf-8", errors="replace")
    has_meta = (
        "fused_ops:" in manifest_text and
        "dispatch_mode=" in manifest_text and
        "output_slot=" in manifest_text and
        "grid=[" in manifest_text
    )
    if not has_meta:
        return fail(f"{manifest.name} is missing expected lowering metadata")
    if "\n  0: MUL " not in manifest_text or "\n  1: ADD " not in manifest_text:
        return fail(f"{manifest.name} should describe one monolithic kernel with MUL feeding ADD")

    dot_path = manifest.with_suffix(".dot")
    if not dot_path.is_file():
        dot_path = shared_artifact_path(manifest, ".dot")
    if not dot_path.is_file():
        return fail(f"{lower_dir} is missing DOT dump for {manifest.name}")
    dot_text = dot_path.read_text(encoding="utf-8", errors="replace")
    if not (
        "LowerCtx IC" in dot_text and
        "Fused Ops" in dot_text and
        "Memory Plan" in dot_text and
        "KOP_ALU" in dot_text and
        "MUL" in dot_text and
        "ADD" in dot_text
    ):
        return fail(f"{dot_path.name} does not show the expected monolithic lowering structure")

    png_path = manifest.with_suffix(".png")
    if not png_path.is_file():
        png_path = shared_artifact_path(manifest, ".png")
    if not png_path.is_file():
        return fail(f"{lower_dir} is missing rendered PNG dump for {manifest.name}")

    if "backend=mtl" in manifest_text:
        msl_path = manifest.with_suffix(".msl")
        if not msl_path.is_file():
            return fail(f"{lower_dir} is missing Metal source dump for {manifest.name}")
        msl = msl_path.read_text(encoding="utf-8", errors="replace")
        if "kernel void K" not in msl or "buf0" not in msl:
            return fail(f"{msl_path.name} does not look like rendered Metal source")

    print(
        "PASS: lower trace kept one monolithic manifest and graph "
        f"({manifest.name}; graph={dot_files[0].name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
