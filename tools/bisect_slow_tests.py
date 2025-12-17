#!/usr/bin/env python
"""Binary-search helper for finding slow pytest tests.

Usage examples:

    # Bisect Kafka unit tests with default settings
    python tools/bisect_slow_tests.py tests/unit/kafka

    # Increase depth / change minimum subset size
    python tools/bisect_slow_tests.py tests/unit/kafka --max-depth 4 --min-size 50

The script uses pytest's --collect-only to gather node IDs, then repeatedly
splits the test list in half, times each half, and recurses into the slower
(or failing) half until the maximum depth or minimum subset size is reached.

It prints a summary of each run so you can see which subsets are slow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class RunResult:
    depth: int
    count: int
    elapsed: float
    returncode: int
    label: str


def run_pytest(node_ids: Sequence[str], quiet: bool = True) -> RunResult:
    """Run pytest for the given node IDs and measure elapsed time.

    The label is a short preview of the subset (first few node IDs).
    """
    if not node_ids:
        return RunResult(depth=0, count=0, elapsed=0.0, returncode=0, label="<empty>")

    cmd: List[str] = [sys.executable, "-m", "pytest"]
    if quiet:
        cmd.append("-q")
    cmd.extend(node_ids)

    label = ", ".join(node_ids[:3])
    start = time.perf_counter()
    proc = subprocess.run(cmd)
    elapsed = time.perf_counter() - start

    return RunResult(
        depth=0,
        count=len(node_ids),
        elapsed=elapsed,
        returncode=proc.returncode,
        label=label,
    )


def collect_node_ids(paths: Sequence[str]) -> List[str]:
    """Collect pytest node IDs for the given paths using --collect-only.

    We filter lines that look like test node IDs (start with "tests/").
    """
    cmd: List[str] = [sys.executable, "-m", "pytest", "--collect-only", "-q", *paths]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[bisect] pytest collection failed", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)

    node_ids: List[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("tests/"):
            node_ids.append(line)

    return node_ids


def bisect_slow_tests(
    node_ids: List[str], max_depth: int, min_size: int
) -> List[RunResult]:
    """Recursively bisect tests, timing subsets and recursing into slower halves.

    Returns a list of RunResult entries for each subset run.
    """
    results: List[RunResult] = []

    def _bisect(current: List[str], depth: int) -> None:
        if depth >= max_depth or len(current) <= min_size:
            result = run_pytest(current)
            result.depth = depth
            results.append(result)
            print(
                f"[bisect] depth={depth} count={result.count} elapsed={result.elapsed:.2f}s "
                f"rc={result.returncode} sample=[{result.label}]",
                flush=True,
            )
            return

        mid = len(current) // 2
        left = current[:mid]
        right = current[mid:]

        left_result = run_pytest(left)
        left_result.depth = depth
        results.append(left_result)
        print(
            f"[bisect] depth={depth} LEFT  count={left_result.count} "
            f"elapsed={left_result.elapsed:.2f}s rc={left_result.returncode} sample=[{left_result.label}]",
            flush=True,
        )

        right_result = run_pytest(right)
        right_result.depth = depth
        results.append(right_result)
        print(
            f"[bisect] depth={depth} RIGHT count={right_result.count} "
            f"elapsed={right_result.elapsed:.2f}s rc={right_result.returncode} sample=[{right_result.label}]",
            flush=True,
        )

        # Prefer failing subset; otherwise recurse into the slower one.
        if left_result.returncode != 0 and right_result.returncode == 0:
            _bisect(left, depth + 1)
        elif right_result.returncode != 0 and left_result.returncode == 0:
            _bisect(right, depth + 1)
        else:
            if left_result.elapsed >= right_result.elapsed:
                _bisect(left, depth + 1)
            else:
                _bisect(right, depth + 1)

    _bisect(node_ids, depth=0)
    return results


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bisect pytest tests to find slow subsets."
    )
    parser.add_argument(
        "paths", nargs="+", help="Test paths to run (e.g. tests/unit/kafka)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=3, help="Maximum bisection depth (default: 3)"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=50,
        help="Minimum subset size before stopping (default: 50)",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    root = Path.cwd()
    print(f"[bisect] running from {root} with paths={args.paths}")

    node_ids = collect_node_ids(args.paths)
    if not node_ids:
        print("[bisect] no tests collected", file=sys.stderr)
        return 1

    print(
        f"[bisect] collected {len(node_ids)} tests; max_depth={args.max_depth} min_size={args.min_size}"
    )
    results = bisect_slow_tests(
        node_ids, max_depth=args.max_depth, min_size=args.min_size
    )

    # Summary of slowest subsets
    sorted_results = sorted(results, key=lambda r: r.elapsed, reverse=True)
    print("\n[bisect] slowest subsets:")
    for result in sorted_results[:10]:
        print(
            f"  depth={result.depth} count={result.count} elapsed={result.elapsed:.2f}s "
            f"rc={result.returncode} sample=[{result.label}]",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
