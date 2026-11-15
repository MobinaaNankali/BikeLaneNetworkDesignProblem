"""Run the bike-lane NDP on many RANDOM candidate sets of a fixed size.

Goal: check whether the variation in computation time is driven by the choice of
candidate set. Each replicate draws a fresh random set of candidate links AND
random build costs (mirroring the original random 1/2/3 instances), solves it,
saves the per-instance result to CSV, and at the end reports the distribution of
computation times across replicates.

This script does NOT modify any existing code. It imports the already-validated
helpers from run_em_highest_flow_candidates.py (whose run code lives under
`if __name__ == "__main__"`, so importing it is side-effect free) and reuses the
same in-place candidate-file writer, so every generated instance has links2 == 0
and runs through the solver without the setY(None) crash.
"""

import argparse
import csv
import random
import statistics
import time
from pathlib import Path

from run_em_highest_flow_candidates import (
    DATA_DIR,
    INFLATE_TRIPS,
    NETWORK_NAME,
    SCAL_FLOW,
    baseline_ue_flows,
    parse_net_file,
    solve_instance,
    write_candidate_file,
)

CANDIDATE_METHOD = "random_uniform"


def percentile(sorted_values, q):
    """Linear-interpolation percentile (q in [0, 100]) over an already-sorted list."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (q / 100.0) * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def summarize(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    return {
        "n": n,
        "min": values[0],
        "p25": percentile(values, 25),
        "median": percentile(values, 50),
        "mean": statistics.fmean(values),
        "p75": percentile(values, 75),
        "max": values[-1],
        "std": statistics.pstdev(values) if n > 1 else 0.0,
    }


def build_replicate_file(instance_name, metadata, links, rng, cost_min, cost_max, size):
    """Pick `size` random links, assign random build costs, and write the instance."""
    # Reset every link's cost so leftover costs from a previous replicate don't leak.
    for link in links:
        link["cost"] = 0.0

    selected = rng.sample(links, size)
    for link in selected:
        link["cost"] = rng.uniform(cost_min, cost_max)

    candidate_ids = [link["original_index"] for link in selected]
    write_candidate_file(DATA_DIR / f"{instance_name}.txt", metadata, links, candidate_ids)
    return selected


def run_replicates(args):
    net_path = DATA_DIR / "net.txt"
    metadata, links = parse_net_file(net_path)

    # One baseline UE solve so we can record the total flow carried by each random set.
    print("--- Computing baseline UE flows on net.txt (once) ---")
    _, flows = baseline_ue_flows()

    headers = [
        "Network",
        "Instance",
        "Candidate_Method",
        "Replicate",
        "Seed",
        "Num_Candidates",
        "Candidate_Flow_Sum",
        "Status",
        "Num_BB_Nodes",
        "Gap_Percent",
        "Objective_UB",
        "Bike_Share",
        "Car_Share",
        "Selected_Links",
        "Selected_link_list",
        "Total_Time_s",
        "TAP_Time_s",
        "CG_Time_s (Pricing)",
        "RMP_Time_s",
        "Candidate_Links",
    ]

    time_records = []  # (replicate, status, total_time)

    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# Parameters"])
        writer.writerow(
            ["network", "size", "replicates", "base_seed", "w", "bike_bonus",
             "budget_percentage", "timelimit", "use_ub_initialization",
             "cost_min", "cost_max", "candidate_method"]
        )
        writer.writerow(
            [NETWORK_NAME, args.size, args.replicates, args.seed, args.w, args.bike_bonus,
             args.budget_percentage, args.timelimit, args.use_ub_initialization,
             args.cost_min, args.cost_max, CANDIDATE_METHOD]
        )
        writer.writerow([])
        writer.writerow(headers)
        f.flush()

        for rep in range(args.replicates):
            seed = args.seed + rep
            rng = random.Random(seed)
            instance_name = f"EM_RANDOM_size{args.size}_rep{rep}"

            selected = build_replicate_file(
                instance_name, metadata, links, rng, args.cost_min, args.cost_max, args.size
            )
            candidate_links_str = ";".join(
                f"({link['init']},{link['term']})" for link in selected
            )
            candidate_flow_sum = sum(flows[link["original_index"]] for link in selected)

            print(f"\n--- RUNNING {NETWORK_NAME} {instance_name} (seed={seed}) ---")
            start = time.time()
            try:
                result = solve_instance(instance_name, args)
                writer.writerow(
                    [
                        NETWORK_NAME,
                        instance_name,
                        CANDIDATE_METHOD,
                        rep,
                        seed,
                        args.size,
                        candidate_flow_sum,
                        result["status"],
                        result["num_bb_nodes"],
                        result["gap"],
                        result["best_ub"],
                        result["bike_share"],
                        result["car_share"],
                        result["selected_links_str"],
                        ";".join(str(link) for link in result["selected_link_list"]),
                        result["total_time"],
                        result["tap_time"],
                        result["pricing_time"],
                        result["rmp_time"],
                        candidate_links_str,
                    ]
                )
                time_records.append((rep, result["status"], result["total_time"]))
                print(
                    f"--- COMPLETED {instance_name} in {time.time() - start:.1f}s: "
                    f"{result['status']} (solver total_time={result['total_time']:.1f}s) ---"
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                writer.writerow(
                    [NETWORK_NAME, instance_name, CANDIDATE_METHOD, rep, seed, args.size,
                     candidate_flow_sum, "ERROR", str(exc)]
                )
                print(f"!!! FAILED {instance_name}: {exc} !!!")
            f.flush()

            if not args.keep_instance_files:
                (DATA_DIR / f"{instance_name}.txt").unlink(missing_ok=True)

    report_distribution(args, time_records)


def report_distribution(args, time_records):
    all_times = [t for (_, _, t) in time_records]
    converged_times = [t for (_, s, t) in time_records if s == "Converged"]

    print("\n" + "=" * 64)
    print(f"COMPUTATION-TIME DISTRIBUTION over {len(time_records)} replicates")
    print(f"(network={NETWORK_NAME}, size={args.size}, method={CANDIDATE_METHOD})")
    print("=" * 64)

    status_counts = {}
    for _, s, _ in time_records:
        status_counts[s] = status_counts.get(s, 0) + 1
    print("Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))

    def print_block(label, values):
        stats = summarize(values)
        if stats is None:
            print(f"\n[{label}] no data")
            return None
        print(f"\n[{label}] Total_Time_s (n={stats['n']})")
        print(f"  min    = {stats['min']:.2f}")
        print(f"  p25    = {stats['p25']:.2f}")
        print(f"  median = {stats['median']:.2f}")
        print(f"  mean   = {stats['mean']:.2f}")
        print(f"  p75    = {stats['p75']:.2f}")
        print(f"  max    = {stats['max']:.2f}")
        print(f"  std    = {stats['std']:.2f}")
        if stats["min"] > 0:
            print(f"  max/min ratio = {stats['max'] / stats['min']:.2f}x")
        return stats

    overall = print_block("ALL replicates", all_times)
    converged = print_block("Converged only", converged_times)

    # Write a compact distribution summary CSV alongside the per-instance results.
    with args.time_summary.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subset", "n", "min", "p25", "median", "mean", "p75", "max", "std"])
        for label, stats in (("all", overall), ("converged", converged)):
            if stats is not None:
                writer.writerow([
                    label, stats["n"], stats["min"], stats["p25"], stats["median"],
                    stats["mean"], stats["p75"], stats["max"], stats["std"],
                ])
    print(f"\nTime-distribution summary written to {args.time_summary}")

    if args.plot and all_times:
        save_plot(args, time_records)


def save_plot(args, time_records):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib missing / no backend
        print(f"(Skipping plot: matplotlib unavailable: {exc})")
        return

    all_times = [t for (_, _, t) in time_records]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.hist(all_times, bins=min(10, len(all_times)), color="#4C78A8", edgecolor="white")
    ax1.set_title(f"Compute time distribution\n{NETWORK_NAME}, size {args.size}, "
                  f"{len(all_times)} random sets")
    ax1.set_xlabel("Total_Time_s")
    ax1.set_ylabel("Replicates")

    ax2.boxplot(all_times, vert=True, showmeans=True)
    ax2.scatter([1] * len(all_times), all_times, alpha=0.6, color="#4C78A8")
    ax2.set_title("Spread across candidate sets")
    ax2.set_ylabel("Total_Time_s")
    ax2.set_xticks([])

    fig.tight_layout()
    fig.savefig(args.plot_output, dpi=130)
    print(f"Distribution plot written to {args.plot_output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the bike-lane NDP on N random candidate sets and report the "
                    "distribution of computation times."
    )
    parser.add_argument("--size", type=int, default=30, help="Candidate-set size (fixed across replicates).")
    parser.add_argument("--replicates", type=int, default=10, help="Number of random candidate sets.")
    parser.add_argument("--seed", type=int, default=20260709, help="Base seed; replicate r uses seed+r.")
    parser.add_argument("--w", type=float, default=0.3)
    parser.add_argument("--bike-bonus", type=float, default=0.4)
    parser.add_argument("--budget-percentage", type=float, default=50.0)
    parser.add_argument("--timelimit", type=float, default=3600 * 4)
    parser.add_argument("--cost-min", type=float, default=0.004)
    parser.add_argument("--cost-max", type=float, default=0.02)
    parser.add_argument("--ub-initialization", dest="use_ub_initialization", action="store_true")
    parser.add_argument("--no-ub-initialization", dest="use_ub_initialization", action="store_false")
    # Output paths default to None here and are filled in with the candidate-set size
    # baked into the filename (see resolve_output_paths), so running a different number
    # of candidates writes to a separate file instead of overwriting the previous run.
    parser.add_argument("--output", type=Path, default=None,
                        help="Per-instance results CSV (default: resultsEM_random_candidates_size<N>.csv).")
    parser.add_argument("--time-summary", type=Path, default=None,
                        help="Time-distribution summary CSV (default: ..._size<N>_time_distribution.csv).")
    parser.add_argument("--plot", dest="plot", action="store_true", help="Save a histogram/boxplot PNG (default on).")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--plot-output", type=Path, default=None,
                        help="Distribution PNG (default: ..._size<N>_time_distribution.png).")
    parser.add_argument("--keep-instance-files", dest="keep_instance_files", action="store_true",
                        help="Keep the generated data/<net>/EM_RANDOM_*.txt files (default on).")
    parser.add_argument("--delete-instance-files", dest="keep_instance_files", action="store_false")
    parser.set_defaults(use_ub_initialization=False, plot=True, keep_instance_files=True)
    return parser.parse_args()


def resolve_output_paths(args):
    """Bake the candidate-set size into any output path the user did not override,
    so different candidate counts save to separate files."""
    if args.output is None:
        args.output = Path(f"resultsEM_random_candidates_size{args.size}.csv")
    if args.time_summary is None:
        args.time_summary = Path(f"resultsEM_random_candidates_size{args.size}_time_distribution.csv")
    if args.plot_output is None:
        args.plot_output = Path(f"resultsEM_random_candidates_size{args.size}_time_distribution.png")
    return args


if __name__ == "__main__":
    args = resolve_output_paths(parse_args())
    run_replicates(args)
    print(f"\nPer-instance results written to {args.output}")
