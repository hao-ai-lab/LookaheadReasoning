import argparse
import json
import os
from typing import List, Tuple


def list_result_files(directory: str) -> List[str]:
    """List JSON result files to analyze in a directory.

    Skips summary files like accuracy.json and results.csv.
    """
    if not os.path.isdir(directory):
        return []
    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        # Only consider .json files; skip known summary files
        if name.lower().endswith(".json") and name.lower() != "accuracy.json":
            files.append(path)
    return sorted(files)


def compute_accept_rate(files: List[str]) -> Tuple[float, int, int]:
    """Compute accept rates from a list of JSON result files.

    Returns a tuple of:
    - overall_accept_rate: aggregated over all accepts across files
    - num_files_used: number of files that contained a non-empty accepts list
    - num_accept_decisions: total number of accept entries aggregated
    """
    total_accept_sum = 0
    total_accept_count = 0
    num_files_used = 0

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        accepts = data.get("accepts")
        # accepts should be a list of 0/1 decisions; skip invalid
        if not isinstance(accepts, list) or len(accepts) == 0:
            continue

        # Filter to numeric-like entries (0/1 or booleans)
        numeric_accepts: List[int] = []
        for v in accepts:
            if isinstance(v, bool):
                numeric_accepts.append(1 if v else 0)
            elif isinstance(v, (int, float)):
                numeric_accepts.append(1 if float(v) >= 0.5 else 0)
            else:
                # Ignore non-numeric entries
                continue

        if len(numeric_accepts) == 0:
            continue

        s = sum(numeric_accepts)
        c = len(numeric_accepts)
        total_accept_sum += s
        total_accept_count += c
        num_files_used += 1

    overall = (total_accept_sum / total_accept_count) if total_accept_count > 0 else 0.0
    return overall, num_files_used, total_accept_count


def analyze_directories(directories: List[str]) -> None:
    """Analyze one or more directories and print accept rate summaries."""
    for d in directories:
        files = list_result_files(d)
        overall, num_files, num_accepts = compute_accept_rate(files)
        print(f"Directory: {d}")
        print(f"  Files analyzed: {num_files}")
        print(f"  Accept decisions: {num_accepts}")
        print(f"  Overall accept rate: {overall:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute average accept rate from result directories (using 'accepts' arrays in JSON files)."
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        help="One or more directories to analyze",
    )
    args = parser.parse_args()

    analyze_directories(args.dirs)


if __name__ == "__main__":
    main()