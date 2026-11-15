"""
Re-run multiple bike bonus experiments with tighter gap threshold
"""

import csv
from src import Network
from src import NewOA_DNDP_withtimelimit

# Configuration - multiple cases to run
cases = [
    {'net': 'EasternMassachusetts', 'ins': 'EM_CNDP_20_3', 'bonus': 0.72},
    {'net': 'EasternMassachusetts', 'ins': 'EM_CNDP_30_3', 'bonus': 0.45},
    {'net': 'EasternMassachusetts', 'ins': 'EM_CNDP_30_3', 'bonus': 0.81},
]

target_gap = 0.005  # 0.5% gap threshold

scal_flow = {'EasternMassachusetts': 1e-3}
inflate_trips = {'EasternMassachusetts': 4}

results = []

for case in cases:
    net = case['net']
    ins = case['ins']
    bonus_val = case['bonus']
    
    print("="*70)
    print(f"RE-RUNNING: {ins} with BIKE BONUS = {bonus_val}")
    print("="*70)
    print(f"Target Gap: {target_gap} ({target_gap*100:.1f}%)")
    print("="*70)

    # Setup
    scale_dem = 1
    b_prop = 0.5
    inflate_cost = 1

    # Initialize network
    network = Network.Network(
        net, ins, b_prop, 1e-0,
        scal_flow[net],
        inflate_trips[net] * scale_dem
    )

    # Create solver
    solver = NewOA_DNDP_withtimelimit.OA_DNDP_CG(
        network,
        inflate_cost,
        use_ub_initialization=False,
        budget_percentage=50.0
    )

    # Set parameters
    solver.w = 1.0
    solver.bikelane_bonus = bonus_val
    solver.timelimit = 3600  # 1 hour max
    solver.params.min_NDP_gap = target_gap  # Set tighter gap threshold (0.5%)

    # Calculate total cost for reference
    total_cost = solver.budget * 2
    num_candidates = len(solver.varlinks)

    print(f"Total possible cost: {total_cost:.4f}")
    print(f"Number of candidate links: {num_candidates}")
    print(f"Budget: {solver.budget:.4f}")
    print("="*70)
    print("\nRunning solver...")

    # Run solver
    (
        best_ub,
        tot_time,
        tap_time,
        pricing_time,
        rmp_time,
        num_bb_nodes,
        gap,
        bike_share,
        car_share,
        selected_links_str,
        selected_links,
        ub_history
    ) = solver.solve()

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Objective: {best_ub:.4f}")
    print(f"Gap: {gap:.6f} ({gap*100:.4f}%)")
    print(f"Bike Share: {bike_share:.4f}")
    print(f"Car Share: {car_share:.4f}")
    print(f"Car/Bike Ratio: {car_share/bike_share:.4f}")
    print(f"Number of BB Nodes: {num_bb_nodes}")
    print(f"Total Time: {tot_time:.2f} seconds")
    print(f"Number of Selected Links: {len(selected_links)}")
    print(f"Selected Links: {[str(link) for link in selected_links]}")
    print("="*70)

    # Save result
    status = 'OPTIMAL' if gap < 1e-3 else 'Converged' if gap < target_gap else 'TIME_LIMIT'
    output_line = f"{net},{ins},{bonus_val},50.0,{solver.budget},{total_cost},{num_bb_nodes},{gap},{best_ub},{bike_share},{car_share},{bike_share},{car_share},{car_share/bike_share},{len(selected_links)},{num_candidates},\"{';'.join([str(link) for link in selected_links])}\",{tot_time},{status}"
    
    print(f"\nCSV Line to replace:")
    print(output_line)
    
    results.append({
        'ins': ins,
        'bonus': bonus_val,
        'line': output_line
    })

print("\n" + "="*70)
print("ALL RESULTS SUMMARY")
print("="*70)
for r in results:
    print(f"\n{r['ins']} - Bike Bonus {r['bonus']}:")
    print(r['line'])
