import argparse
import csv
import random
import time
from pathlib import Path

from src import Network


NETWORK_NAME = "EasternMassachusetts"
DATA_DIR = Path("data") / NETWORK_NAME

SCAL_FLOW = {
    "SiouxFalls": 1e-3,
    "EasternMassachusetts": 1e-3,
    "BerlinMitteCenter": 1e-3,
    "Anaheim": 1e-3,
    "Barcelona": 1e-3,
    "Braess": 1,
    "HarkerFriesz": 1,
}

INFLATE_TRIPS = {
    "SiouxFalls": 1,
    "EasternMassachusetts": 4,
    "BerlinMitteCenter": 2,
    "Anaheim": 4,
    "Barcelona": 2,
    "Braess": 1,
    "HarkerFriesz": 0.25,
}

INSTANCE_SIZES = (10, 20, 30)


def parse_net_file(path):
    metadata = []
    data_header = None
    links = []
    in_data = False

    with path.open() as f:
        for raw in f:
            stripped = raw.strip()
            if not in_data:
                if stripped.startswith("~"):
                    data_header = "~\tinit_node\tterm_node\tcapacity\tlength\tfree_flow_time\tb\tpower\tspeed\ttoll\tlink_type\tcost\t;"
                    in_data = True
                elif stripped:
                    metadata.append(stripped)
                continue

            if not stripped or stripped.startswith("~"):
                continue

            parts = stripped.replace(";", " ").split()
            if len(parts) < 10:
                continue

            links.append(
                {
                    "init": int(parts[0]),
                    "term": int(parts[1]),
                    "capacity": float(parts[2]),
                    "length": float(parts[3]),
                    "free_flow_time": float(parts[4]),
                    "b": float(parts[5]),
                    "power": float(parts[6]),
                    "speed": float(parts[7]),
                    "toll": float(parts[8]),
                    "link_type": int(float(parts[9])),
                    "cost": 0.0,
                    "original_index": len(links),
                }
            )

    if data_header is None:
        raise ValueError(f"Could not find link table header in {path}")

    return metadata, links


def format_number(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.12g}"


def write_candidate_file(path, metadata, links, candidate_ids):
    # Candidates are EXISTING links that receive a nonzero build cost (the solver
    # treats every link with cost > 0 as a bike-lane variable via `varlinks`).
    # We must NOT append them as "new links": readNetwork would put the extra rows
    # in `links2`, and `tapas("UE", None)` -> `setY(None)` crashes iterating a
    # non-empty `links2`. So we keep every link in its original position and only
    # flip the cost column for the selected candidates, and declare zero new links.
    candidate_set = set(candidate_ids)

    with path.open("w", newline="") as f:
        for line in metadata:
            if line.startswith("<NUMBER OF LINKS>"):
                f.write(f"<NUMBER OF LINKS> {len(links)}\n")
            elif line.startswith("<NUMBER OF NEW LINKS>"):
                continue
            elif line.startswith("<END OF METADATA>"):
                f.write("<NUMBER OF NEW LINKS> 0\n")
                f.write("<END OF METADATA>\n\n\n")
            elif line.startswith("<ORIGINAL HEADER>"):
                continue
            else:
                f.write(f"{line}\n")

        f.write("~\tinit_node\tterm_node\tcapacity\tlength\tfree_flow_time\tb\tpower\tspeed\ttoll\tlink_type\tcost\t;\n")
        for link in links:
            cost = link["cost"] if link["original_index"] in candidate_set else 0.0
            row = [
                link["init"],
                link["term"],
                link["capacity"],
                link["length"],
                link["free_flow_time"],
                link["b"],
                link["power"],
                link["speed"],
                link["toll"],
                link["link_type"],
                cost,
            ]
            f.write("\t".join(format_number(v) for v in row) + "\t;\n")


def baseline_ue_flows():
    network = Network.Network(
        NETWORK_NAME,
        "net",
        0.5,
        1e-0,
        SCAL_FLOW[NETWORK_NAME],
        INFLATE_TRIPS[NETWORK_NAME],
    )
    network.tapas("UE", None)
    flows = {link.id: link.x for link in network.links}
    return network, flows


def generate_highest_flow_instances(seed, output_summary):
    net_path = DATA_DIR / "net.txt"
    metadata, links = parse_net_file(net_path)
    baseline_network, flows = baseline_ue_flows()

    ranked = sorted(links, key=lambda link: flows[link["original_index"]], reverse=True)
    rng = random.Random(seed)

    # Assign costs once to the top 30, so the 10/20/30 instances are nested and comparable.
    for link in ranked[: max(INSTANCE_SIZES)]:
        link["cost"] = rng.uniform(0.004, 0.02)

    summary_rows = []
    for size in INSTANCE_SIZES:
        selected = ranked[:size]
        candidate_ids = [link["original_index"] for link in selected]
        instance_name = f"EM_CNDP_{size}_highestflow"
        write_candidate_file(DATA_DIR / f"{instance_name}.txt", metadata, links, candidate_ids)

        for rank, link in enumerate(selected, start=1):
            summary_rows.append(
                [
                    instance_name,
                    rank,
                    link["original_index"],
                    link["init"],
                    link["term"],
                    flows[link["original_index"]],
                    link["length"],
                    link["cost"],
                    link["cost"] * link["length"],
                ]
            )

    with output_summary.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Instance",
                "Flow_Rank",
                "Original_Link_Index",
                "Init_Node",
                "Term_Node",
                "Baseline_UE_Flow",
                "Length",
                "Unit_Cost",
                "Build_Cost",
            ]
        )
        writer.writerows(summary_rows)

    return baseline_network, ranked[: max(INSTANCE_SIZES)]


def solve_instance(instance_name, args):
    from src import NewOA_DNDP_withtimelimit

    network = Network.Network(
        NETWORK_NAME,
        instance_name,
        0.5,
        1e-0,
        SCAL_FLOW[NETWORK_NAME],
        INFLATE_TRIPS[NETWORK_NAME],
    )

    solver = NewOA_DNDP_withtimelimit.OA_DNDP_CG(
        network,
        1,
        use_ub_initialization=args.use_ub_initialization,
        budget_percentage=args.budget_percentage,
    )
    solver.w = args.w
    solver.bikelane_bonus = args.bike_bonus
    solver.timelimit = args.timelimit

    result = solver.solve()
    if len(result) == 12:
        (
            best_ub,
            total_time,
            tap_time,
            pricing_time,
            rmp_time,
            num_bb_nodes,
            gap,
            bike_share,
            car_share,
            selected_links_str,
            selected_link_list,
            ub_history,
        ) = result
    else:
        (
            best_ub,
            total_time,
            tap_time,
            pricing_time,
            rmp_time,
            num_bb_nodes,
            gap,
            bike_share,
            car_share,
            selected_links_str,
            selected_link_list,
        ) = result
        ub_history = []

    if total_time >= solver.timelimit:
        status = "TimeLimit"
    elif gap < solver.params.min_NDP_gap:
        status = "Converged"
    else:
        status = "TreeExhausted"

    return {
        "status": status,
        "best_ub": best_ub,
        "total_time": total_time,
        "tap_time": tap_time,
        "pricing_time": pricing_time,
        "rmp_time": rmp_time,
        "num_bb_nodes": num_bb_nodes,
        "gap": gap,
        "bike_share": bike_share,
        "car_share": car_share,
        "selected_links_str": selected_links_str,
        "selected_link_list": selected_link_list,
        "ub_history": ub_history,
    }


def run_experiments(args):
    headers = [
        "Network",
        "Instance",
        "Candidate_Method",
        "Num_Candidates",
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
    ]

    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# Parameters"])
        writer.writerow(["w", "bike_bonus", "budget_percentage", "timelimit", "use_ub_initialization", "candidate_method"])
        writer.writerow([args.w, args.bike_bonus, args.budget_percentage, args.timelimit, args.use_ub_initialization, "baseline_ue_highest_flow"])
        writer.writerow([])
        writer.writerow(headers)
        f.flush()

        for size in INSTANCE_SIZES:
            instance_name = f"EM_CNDP_{size}_highestflow"
            print(f"\n--- RUNNING {NETWORK_NAME} {instance_name} ---")
            start = time.time()
            try:
                result = solve_instance(instance_name, args)
                writer.writerow(
                    [
                        NETWORK_NAME,
                        instance_name,
                        "baseline_ue_highest_flow",
                        size,
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
                    ]
                )
                print(f"--- COMPLETED {instance_name} in {time.time() - start:.1f}s: {result['status']} ---")
            except Exception as exc:
                import traceback

                traceback.print_exc()
                writer.writerow([NETWORK_NAME, instance_name, "baseline_ue_highest_flow", size, "ERROR", str(exc)])
                print(f"!!! FAILED {instance_name}: {exc} !!!")
            f.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and run EM highest-baseline-UE-flow bike lane candidates.")
    parser.add_argument("--output", type=Path, default=Path("resultsEM_highestflow_candidates.csv"))
    parser.add_argument("--candidate-summary", type=Path, default=Path("em_highestflow_candidate_summary.csv"))
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--w", type=float, default=0.3)
    parser.add_argument("--bike-bonus", type=float, default=0.4)
    parser.add_argument("--budget-percentage", type=float, default=50.0)
    parser.add_argument("--timelimit", type=float, default=3600 * 4)
    parser.add_argument("--ub-initialization", dest="use_ub_initialization", action="store_true")
    parser.add_argument("--no-ub-initialization", dest="use_ub_initialization", action="store_false")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.set_defaults(use_ub_initialization=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_generate:
        generate_highest_flow_instances(args.seed, args.candidate_summary)
        print(f"Candidate files written under {DATA_DIR}")
        print(f"Candidate summary written to {args.candidate_summary}")

    if not args.generate_only:
        run_experiments(args)
        print(f"Results written to {args.output}")
