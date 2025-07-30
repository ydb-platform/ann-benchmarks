#!/usr/bin/env python3
import argparse
import numpy as np

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.utils import compute_metrics_all_runs
from ann_benchmarks.results import get_unique_algorithms, load_all_results


def create_table(all_results, output_file):
    """Create a table with algorithm performance metrics."""

    # Collect data for each algorithm
    table_data = []

    for result in all_results:
        # Extract the metrics we want
        row = {
            "algorithm": result["parameters"],  # Algorithm name with parameters
            "recall": result.get("k-nn", 0.0),  # k-nn metric as recall
            "qps": result.get("qps", 0.0),
            "p50": result.get("p50", 0.0),
            "p95": result.get("p95", 0.0),
            "p99": result.get("p99", 0.0),
            "p999": result.get("p999", 0.0)
        }
        table_data.append(row)

    # Group by algorithm, then sort by recall within each group
    # Extract base algorithm name (part before first parenthesis)
    for row in table_data:
        algo_base = row["algorithm"].split("(")[0] if "(" in row["algorithm"] else row["algorithm"]
        row["algo_base"] = algo_base

    # Sort by algorithm name first, then by recall (descending) and QPS (descending) within each algorithm
    table_data.sort(key=lambda x: (x["algo_base"], -x["recall"], -x["qps"]))

    # Find the longest algorithm name and set column width
    max_algo_len = max(len(row["algorithm"]) for row in table_data) if table_data else 10
    algo_width = max(max_algo_len + 2, 10)  # Add padding, minimum 10 chars

    # Output format
    if output_file:
        output_lines = []

    # Print header
    header = f"{'Algorithm':<{algo_width}} {'Recall':>10} {'QPS':>12} {'P50':>8} {'P95':>8} {'P99':>8} {'P999':>8}"
    print(header)
    print("-" * len(header))

    if output_file:
        output_lines.append(header)
        output_lines.append("-" * len(header))

    # Print data rows
    for row in table_data:
        line = f"{row['algorithm']:<{algo_width}} {row['recall']:>10.4f} {row['qps']:>12.1f} {row['p50']:>8.1f} {row['p95']:>8.1f} {row['p99']:>8.1f} {row['p999']:>8.1f}"
        print(line)
        if output_file:
            output_lines.append(line)

    # Write to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\nTable written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", metavar="DATASET", default="glove-100-angular")
    parser.add_argument("--count", default=10)
    parser.add_argument(
        "--definitions", metavar="FILE", help="load algorithm definitions from FILE", default="algos.yaml"
    )
    parser.add_argument("--limit", default=-1)
    parser.add_argument("-o", "--output", help="Output file path (optional, prints to stdout if not specified)")
    parser.add_argument("--batch", help="Process runs in batch mode", action="store_true")
    parser.add_argument("--recompute", help="Clears the cache and recomputes the metrics", action="store_true")
    args = parser.parse_args()

    dataset, _ = get_dataset(args.dataset)
    count = int(args.count)
    results = load_all_results(args.dataset, count, args.batch)

    if not results:
        raise Exception("No results found for dataset")

    # Compute all metrics for all runs
    all_results = list(compute_metrics_all_runs(dataset, results, args.recompute))

    if not all_results:
        raise Exception("Nothing to process")

    create_table(all_results, args.output)