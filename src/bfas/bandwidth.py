"""Measure the constraint instead of quoting it.

Spec-sheet bandwidth is a ceiling nobody reaches. Every number this project
reports is measured: a STREAM-style sweep gives achievable bandwidth for the
host, and an optional torch read benchmark gives it for whatever accelerator
is present. Run this before anything else, on every machine you report on.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

GB = 1024.0 ** 3


@dataclass
class DeviceProfile:
    """What the hardware can actually do, as measured here."""

    name: str
    device: str
    achievable_gbs: float
    method: str
    working_set_mb: float
    repeats: int
    notes: str = ""

    def to_json(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(asdict(self), handle, indent=2)
        print(f"wrote {path}")

    @staticmethod
    def from_json(path: str) -> "DeviceProfile":
        with open(path) as handle:
            return DeviceProfile(**json.load(handle))


def _triad(n_elements: int, repeats: int) -> float:
    """STREAM-style add: a = b + c_scaled, with the scaling hoisted out of the
    timed region so numpy allocates no temporary inside it.

    Exactly three arrays are touched per element (two read, one written), so
    traffic per pass is 3 * n * itemsize bytes. Counting the traffic exactly is
    the whole point: a kernel with a hidden temporary silently understates
    achievable bandwidth.
    """
    a = np.zeros(n_elements, dtype=np.float32)
    b = np.ones(n_elements, dtype=np.float32)
    c_scaled = np.full(n_elements, 6.0, dtype=np.float32)
    np.add(b, c_scaled, out=a)  # warm the pages

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        np.add(b, c_scaled, out=a)
        best = min(best, time.perf_counter() - start)
    bytes_moved = 3.0 * n_elements * a.itemsize
    return bytes_moved / best / 1e9


def measure_host_bandwidth(
    working_set_mb: float = 512.0, repeats: int = 7
) -> DeviceProfile:
    """Achievable host memory bandwidth in GB/s.

    Keep the working set far larger than last-level cache or the number you get
    is a cache bandwidth, which is both wrong and flattering.
    """
    n_elements = int(working_set_mb * 1024 * 1024 / 4)
    gbs = _triad(n_elements, repeats)
    return DeviceProfile(
        name=platform.processor() or platform.machine(),
        device="cpu",
        achievable_gbs=gbs,
        method="numpy STREAM triad (best of n)",
        working_set_mb=working_set_mb,
        repeats=repeats,
        notes="scale threads with OMP_NUM_THREADS to emulate narrower memory systems",
    )


def measure_torch_bandwidth(
    device: str = "cuda", working_set_mb: float = 1024.0, repeats: int = 20
) -> Optional[DeviceProfile]:
    """Achievable accelerator bandwidth, measured with a read-dominated kernel.

    Returns None when the device is unavailable so callers can skip it without
    branching on hardware.
    """
    try:
        import torch
    except ImportError:
        return None

    if device.startswith("cuda") and not torch.cuda.is_available():
        return None
    if device == "mps" and not getattr(torch.backends, "mps", None):
        return None

    n_elements = int(working_set_mb * 1024 * 1024 / 2)
    x = torch.empty(n_elements, dtype=torch.float16, device=device).fill_(1.0)

    def sync() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    for _ in range(3):
        x.sum()
    sync()

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        x.sum()
        sync()
        best = min(best, time.perf_counter() - start)

    gbs = (n_elements * x.element_size()) / best / 1e9
    name = (
        torch.cuda.get_device_name(0)
        if device.startswith("cuda")
        else device
    )
    return DeviceProfile(
        name=name,
        device=device,
        achievable_gbs=gbs,
        method="torch fp16 full-tensor reduction (best of n)",
        working_set_mb=working_set_mb,
        repeats=repeats,
    )


def sweep_working_sets(sizes_mb: Optional[List[float]] = None) -> List[Dict[str, float]]:
    """Bandwidth against working-set size. The knee in this curve is where you
    fall out of cache, and it tells you whether a measurement is trustworthy."""
    sizes_mb = sizes_mb or [1, 4, 16, 64, 256, 1024]
    rows = []
    for size in sizes_mb:
        n_elements = int(size * 1024 * 1024 / 4)
        rows.append({"working_set_mb": size, "achievable_gbs": _triad(n_elements, 5)})
    return rows


# Reference points for the regimes the paper contrasts. These are published
# figures used only for orientation; every claim in the paper comes from a
# measurement on a machine you ran on.
BANDWIDTH_REGIMES = {
    "mobile-soc-low": 50.0,
    "mobile-soc-high": 90.0,
    "laptop-ddr5": 100.0,
    "workstation-gpu": 700.0,
    "datacenter-hbm": 3350.0,
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Profile achievable memory bandwidth")
    parser.add_argument("--working-set-mb", type=float, default=512.0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--sweep", action="store_true", help="also sweep working-set size")
    parser.add_argument("--torch-device", default=None, help="e.g. cuda, mps")
    parser.add_argument("--out", default="results/device_profile.json")
    args = parser.parse_args()

    profile = measure_host_bandwidth(args.working_set_mb, args.repeats)
    print(f"host: {profile.name}")
    print(f"achievable bandwidth: {profile.achievable_gbs:.1f} GB/s ({profile.method})")

    for regime, value in BANDWIDTH_REGIMES.items():
        marker = " <-- you are here" if 0.7 * value <= profile.achievable_gbs <= 1.4 * value else ""
        print(f"  {regime:<18} {value:>7.0f} GB/s{marker}")

    if args.sweep:
        print("\nworking-set sweep:")
        for row in sweep_working_sets():
            print(f"  {row['working_set_mb']:>7.0f} MB  {row['achievable_gbs']:>7.1f} GB/s")

    if args.torch_device:
        accel = measure_torch_bandwidth(args.torch_device)
        if accel is None:
            print(f"\n{args.torch_device} not available, skipped")
        else:
            print(f"\n{accel.device} ({accel.name}): {accel.achievable_gbs:.1f} GB/s")
            ratio = accel.achievable_gbs / profile.achievable_gbs
            print(f"accelerator is {ratio:.0f}x the host memory system")

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    profile.to_json(args.out)


if __name__ == "__main__":
    main()
