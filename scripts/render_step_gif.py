#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render step_*.png frames into an animated GIF."
    )
    parser.add_argument("step_dir", type=Path, help="Directory containing step_*.png frames.")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output GIF path. Defaults to <step_dir>/step_trace.gif.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.25,
        help="Frame rate for the GIF animation.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional output width in pixels. Height is preserved.",
    )
    return parser.parse_args()


def sorted_frames(step_dir: Path) -> list[Path]:
    return sorted(p for p in step_dir.glob("step_*.png") if p.is_file())


def magick_identify(magick: str, frames: list[Path]) -> list[tuple[int, int]]:
    if Path(magick).name == "magick":
        cmd = [magick, "identify", "-format", "%w %h\n", *map(str, frames)]
    else:
        identify = shutil.which("identify")
        if not identify:
            raise SystemExit("need `identify` with ImageMagick when only `convert` is available")
        cmd = [identify, "-format", "%w %h\n", *map(str, frames)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    dims: list[tuple[int, int]] = []
    for line in out.splitlines():
        w, h = line.strip().split()
        dims.append((int(w), int(h)))
    return dims


def ffmpeg_render(ffmpeg: str, step_dir: Path, output: Path, fps: float, width: int) -> None:
    pattern = str(step_dir / "step_*.png")
    scale = f"scale={width}:-1:flags=lanczos," if width > 0 else ""
    with tempfile.TemporaryDirectory(prefix="thvm_step_gif_") as tmpdir:
        palette = Path(tmpdir) / "palette.png"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-pattern_type",
                "glob",
                "-i",
                pattern,
                "-vf",
                f"{scale}palettegen",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-pattern_type",
                "glob",
                "-i",
                pattern,
                "-i",
                str(palette),
                "-lavfi",
                f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
                str(output),
            ],
            check=True,
        )


def magick_render(magick: str, frames: list[Path], output: Path, fps: float, width: int) -> None:
    delay = max(1, int(round(100.0 / fps)))
    dims = magick_identify(magick, frames)
    if width > 0:
        scaled = [(width, max(1, round(h * width / w))) for w, h in dims]
        max_w = width
        max_h = max(h for _, h in scaled)
    else:
        max_w = max(w for w, _ in dims)
        max_h = max(h for _, h in dims)
    with tempfile.TemporaryDirectory(prefix="thvm_step_gif_pad_") as tmpdir:
        padded: list[Path] = []
        for i, frame in enumerate(frames):
            out = Path(tmpdir) / f"frame_{i:04d}.png"
            cmd = [magick, str(frame)]
            if width > 0:
                cmd += ["-resize", f"{width}x"]
            cmd += ["-background", "white",
                    "-gravity", "center",
                    "-extent", f"{max_w}x{max_h}",
                    str(out)]
            subprocess.run(cmd, check=True)
            padded.append(out)
        cmd = [magick, "-delay", str(delay), "-loop", "0", *map(str, padded), str(output)]
        subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    step_dir = args.step_dir
    if not step_dir.is_dir():
        raise SystemExit(f"{step_dir} is not a directory")
    frames = sorted_frames(step_dir)
    if not frames:
        raise SystemExit(f"no step_*.png frames found in {step_dir}")

    output = args.output or step_dir / "step_trace.gif"
    output.parent.mkdir(parents=True, exist_ok=True)

    magick = shutil.which("magick") or shutil.which("convert")
    ffmpeg = shutil.which("ffmpeg")
    if magick:
        magick_render(magick, frames, output, args.fps, args.width)
    elif ffmpeg:
        ffmpeg_render(ffmpeg, step_dir, output, args.fps, args.width)
    else:
        raise SystemExit("need ffmpeg or ImageMagick (`magick`/`convert`) on PATH")

    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
