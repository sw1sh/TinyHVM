"""Mandelbrot set via tinygrad — reference for the TinyHVM port.

Run:
    METAL=1 python examples/mandelbrot/mandelbrot_tinygrad.py
"""
from __future__ import annotations
import sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tinygrad"))

from tinygrad import Tensor


def mandelbrot(H: int, W: int, N_ITER: int,
               x0: float = -2.0, x1: float = 1.0,
               y0: float = -1.2, y1: float = 1.2) -> Tensor:
    cr = Tensor.linspace(x0, x1, W).reshape(1, W).expand(H, W)
    ci = Tensor.linspace(y0, y1, H).reshape(H, 1).expand(H, W)

    zr = Tensor.zeros(H, W).contiguous()
    zi = Tensor.zeros(H, W).contiguous()
    cnt = Tensor.zeros(H, W).contiguous()

    for _ in range(N_ITER):
        zr2 = zr * zr
        zi2 = zi * zi
        alive = (zr2 + zi2 < 4.0).float()
        zr, zi = zr2 - zi2 + cr, (zr * zi) * 2.0 + ci
        cnt = cnt + alive

    return cnt


def to_ppm(flat: list[float], H: int, W: int, path: str, n_iter: int) -> None:
    with open(path, "wb") as f:
        f.write(f"P6\n{W} {H}\n255\n".encode())
        for v in flat:
            t = v / n_iter
            r = int(9 * (1 - t) * t * t * t * 255)
            g = int(15 * (1 - t) ** 2 * t * t * 255)
            b = int(8.5 * (1 - t) ** 3 * t * 255)
            f.write(bytes((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))))


if __name__ == "__main__":
    H, W, N = 512, 768, 100

    for _ in range(2):
        t0 = time.perf_counter()
        out = mandelbrot(H, W, N).realize()
        dt = time.perf_counter() - t0
        print(f"mandel {H}x{W} × {N} iter: {dt*1000:.1f} ms")

    flat_nested = out.tolist()
    flat = [v for row in flat_nested for v in row]
    out_path = os.path.join(HERE, "mandelbrot_tinygrad.ppm")
    to_ppm(flat, H, W, out_path, N)
    print(f"wrote {out_path}  range=[{min(flat):.0f}, {max(flat):.0f}]")
