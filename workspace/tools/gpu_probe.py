# -*- coding: utf-8 -*-
"""
GPU capability probe for LFM. Prints hardware limits and sizing guidance.

Usage (Windows):
  py -3 workspace\tools\gpu_probe.py
  or
  python workspace/tools/gpu_probe.py

This prints:
- GPU name, compute capability, SM count
- VRAM total/free
- Shared memory per block/SM, warp size, max threads per block
- Estimated memory bandwidth (if available)
- CuPy version
- Memory fit estimates for 128^3 .. 512^3 in FP32 with headroom
- Shared-memory friendly block tile recommendations
"""
from __future__ import annotations
import sys
from pathlib import Path


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:0.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def main() -> int:
    try:
        import cupy as cp
        from cupy.cuda import runtime as cu_rt
    except Exception as e:
        print("CuPy not available or no CUDA runtime. Install CuPy and a CUDA-capable driver.", file=sys.stderr)
        print(f"Detail: {e}", file=sys.stderr)
        return 1

    dev_id = cp.cuda.Device().id
    props = cu_rt.getDeviceProperties(dev_id)
    name = props["name"].decode() if isinstance(props["name"], (bytes, bytearray)) else props["name"]
    sm_maj, sm_min = props["major"], props["minor"]
    sm = f"{sm_maj}.{sm_min}"
    warp_size = props["warpSize"]
    max_threads_per_block = props["maxThreadsPerBlock"]
    sm_count = props["multiProcessorCount"]
    shared_per_block = props["sharedMemPerBlock"]  # bytes
    shared_per_sm = getattr(props, "sharedMemPerMultiprocessor", 0) or 0
    regs_per_block = props["regsPerBlock"]
    l2_bytes = getattr(props, "l2CacheSize", 0) or 0

    free_b, total_b = cu_rt.memGetInfo()

    mem_clock_khz = getattr(props, "memoryClockRate", 0)  # kHz
    bus_width_bits = getattr(props, "memoryBusWidth", 0)  # bits
    bw_gbs = (mem_clock_khz * 1000 * (bus_width_bits / 8) * 2) / 1e9 if (mem_clock_khz and bus_width_bits) else 0.0

    print("GPU Probe")
    print("=" * 60)
    print(f"Device            : {name}")
    print(f"Compute Capability: {sm}")
    print(f"SM count          : {sm_count}")
    print(f"Warp size         : {warp_size}")
    print(f"Max threads/block : {max_threads_per_block}")
    print(f"Shared mem/block  : {_human(shared_per_block)}")
    print(f"Shared mem/SM     : {_human(shared_per_sm)}")
    print(f"Regs per block    : {regs_per_block}")
    print(f"L2 cache          : {_human(l2_bytes)}")
    print(f"VRAM free/total   : {_human(free_b)} / {_human(total_b)}")
    print(f"Est. mem bandwidth: {bw_gbs:.1f} GB/s" if bw_gbs else "Est. mem bandwidth: (n/a)")
    print(f"CuPy version      : {cp.__version__}")
    print()

    # Memory model assumptions (float32)
    # Arrays: E, E_prev, chi, lap_scratch, E_next (5) → 20 bytes/cell
    # Safety factor 1.4 for temps/masks → ~28 bytes/cell
    bytes_per_cell = 28

    def fit(n):
        cells = n ** 3
        need = cells * bytes_per_cell
        return need, (need < free_b * 0.85)

    print("Estimated FP32 memory needs (5 arrays + 40% headroom):")
    for n in (128, 192, 256, 320, 384, 512):
        need, ok = fit(n)
        print(f"- {n:>3}^3: {_human(need)}  {'OK' if ok else 'NO'}")
    print()

    # Tile/block recommendations for 7-point stencil with 1-cell halo in shared mem
    # Shared tile bytes ~ (tx+2)*(ty+2)*(tz+2) * 4B (one field)
    def pick_tile(shared_limit=shared_per_block, per_field=4, fields=1):
        candidates = [(8, 8, 8), (16, 8, 8), (8, 16, 8), (8, 8, 16), (12, 8, 8), (8, 12, 8)]
        recs = []
        for tx, ty, tz in candidates:
            sx, sy, sz = tx + 2, ty + 2, tz + 2
            bytes_tile = sx * sy * sz * per_field * fields
            if bytes_tile <= shared_limit:
                threads = tx * ty * tz
                recs.append((threads, tx, ty, tz, bytes_tile))
        recs.sort(reverse=True)
        return recs[:4]

    print("Recommended thread-block tiles (shared-mem friendly):")
    for threads, tx, ty, tz, b in pick_tile(fields=1):
        print(f"- block=({tx},{ty},{tz})  threads={threads:>4}  shared≈{_human(b)}")
    print("\nNotes:")
    print("- Use float32 for main update; compute energy in float64 with compensated (Kahan) sum.")
    print("- Start with block (8,8,8) or (16,8,8); profile occupancy and bandwidth.")
    print("- For 256^3 and larger, prefer fused Laplacian+Verlet kernel to cut global memory traffic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
