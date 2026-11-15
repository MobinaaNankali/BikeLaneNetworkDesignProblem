"""
Sensitivity Analysis: Varying Budget
Analyzes how budget percentage affects bike lane selection and performance
"""

import csv
import json
import numpy as np
from src import Network
from src import NewOA_DNDP_withtimelimit

# Configuration
SELECTED_INSTANCES = {
    'EasternMassachusetts': [
        'EM_CNDP_20_3',  # 20 candidates
        'EM_CNDP_30_3'   # 30 candidates
    ]
}

scal_flow = {'EasternMassachusetts': 1e-3}
inflate_trips = {'EasternMassachusetts': 4}

def calculate_budget_percentages(n_points=9):
    """
    Calculate budget percentages to test.
    Range from 10% to 90% in increments of 10%
    """
    budget_percentages = np.linspace(10, 90, n_points)  # 10%, 20%, ..., 90%
    return budget_percentages

def run_budget_sensitivity_analysis():
    """
    Run experiments with varying budget percentages
    """
    
    # Calculate budget percentages to test
    budget_percentages = calculate_budget_percentages(n_points=9)
    
    print("="*70)
    print("SENSITIVITY ANALYSIS: Budget Percentage")
    print("="*70)
    print(f"\nTesting {len(budget_percentages)} different budget levels:")
    print(f"Budget range: {budget_percentages[0]:.0f}% to {budget_percentages[-1]:.0f}%")
    print(f"Budget values: {[f'{b:.0f}%' for b in budget_percentages]}")
    print("\nFixed parameters:")
    print("  - w = 1.0 (equal weight for car and bike)")
    print("  - No UB initialization (for consistency)")
    print("="*70)
    
    # Results storage
    results = []
    detailed_results = {}
    
    # CSV output
    output_filename = 'sensitivity_budget_results.csv'
    headers = [
        'Network',
        'Instance',
        'Budget_Percent',
        'Budget_Value',
        'Total_Cost',
        'Num_BB_Nodes',
        'Gap_Percent',
        'Objective_Total',
        'Bike_Share',
        'Car_Share',
        'Car_to_Bike_Ratio',
        'Num_Selected_Links',
        'Max_Possible_Links',
        'Selected_Links',
        'Total_Time_s',
        'Status'
    ]
    
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        # Loop through selected instances
        for net, instances in SELECTED_INSTANCES.items():
            for ins in instances:
                print(f"\n{'='*70}")
                print(f"Instance: {net} - {ins}")
                print(f"{'='*70}")
                
                instance_results = {}
                
                # First, calculate total cost for this instance
                # We need to initialize once to get the total cost
                network_temp = Network.Network(
                    net, ins, 0.5, 1e-0,
                    scal_flow[net],
                    inflate_trips[net]
                )
                solver_temp = NewOA_DNDP_withtimelimit.OA_DNDP_CG(
                    network_temp, 1, use_ub_initialization=False, budget_percentage=50.0
                )
                total_cost = solver_temp.budget * 2  # Since default is 50%
                num_candidates = len(solver_temp.varlinks)
                
                print(f"Total possible cost: {total_cost:.2f}")
                print(f"Number of candidate links: {num_candidates}")
                print(f"{'='*70}\n")
                
                # Loop through budget percentages
                for budget_pct in budget_percentages:
                    print(f"  Testing Budget = {budget_pct:.0f}% ({budget_pct/100 * total_cost:.2f})")
                    print(f"  {'-'*60}")
                    
                    try:
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
                        
                        # Create solver with specified budget percentage
                        solver = NewOA_DNDP_withtimelimit.OA_DNDP_CG(
                            network,
                            inflate_cost,
                            use_ub_initialization=False,
                            budget_percentage=budget_pct  # Pass budget percentage here
                        )
                        
                        # Keep w = 1 (equal weight)
                        solver.w = 1.0
                        
                        # Shorter time limit for sensitivity analysis
                        solver.timelimit = 1800  # 30 minutes max per run
                        
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
                            selected_links,
                            selected_link_list,
                            ub_history
                        ) = solver.solve()
                        
                        # Determine status
                        if tot_time >= solver.timelimit:
                            status = 'TimeLimit'
                        elif gap < solver.params.min_NDP_gap:
                            status = 'Converged'
                        else:
                            status = 'TreeExhausted'
                        
                        # Calculate car to bike ratio
                        if bike_share > 0:
                            car_to_bike_ratio = car_share / bike_share
                        else:
                            car_to_bike_ratio = float('inf')
                        
                        # Count selected links
                        num_selected = len(selected_link_list)
                        
                        print(f"    Status: {status}")
                        print(f"    BB Nodes: {num_bb_nodes}")
                        print(f"    Bike Share: {bike_share:.2f}")
                        print(f"    Car Share: {car_share:.2f}")
                        print(f"    Car/Bike Ratio: {car_to_bike_ratio:.4f}")
                        print(f"    Selected Links: {num_selected}/{num_candidates}")
                        print(f"    Time: {tot_time:.2f}s")
                        
                        # Store results
                        result_row = [
                            net,
                            ins,
                            budget_pct,
                            solver.budget,
                            total_cost,
                            num_bb_nodes,
                            gap,
                            best_ub,
                            bike_share,
                            car_share,
                            car_to_bike_ratio,
                            num_selected,
                            num_candidates,
                            ';'.join(selected_link_list),
                            tot_time,
                            status
                        ]
                        
                        writer.writerow(result_row)
                        csvfile.flush()
                        
                        # Store for JSON
                        instance_results[f"budget_{budget_pct:.0f}pct"] = {
                            'budget_percent': budget_pct,
                            'budget_value': solver.budget,
                            'total_cost': total_cost,
                            'num_bb_nodes': num_bb_nodes,
                            'gap': gap,
                            'objective': best_ub,
                            'bike_share': bike_share,
                            'car_share': car_share,
                            'car_to_bike_ratio': car_to_bike_ratio,
                            'selected_links': selected_link_list,
                            'num_selected': num_selected,
                            'max_possible': num_candidates,
                            'total_time': tot_time,
                            'status': status,
                            'ub_history': ub_history
                        }
                        
                    except Exception as e:
                        print(f"    ERROR: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        
                        writer.writerow([net, ins, budget_pct, 'ERROR', 'ERROR', 'ERROR', str(e)])
                        csvfile.flush()
                
                # Store instance results
                detailed_results[f"{net}_{ins}"] = instance_results
    
    # Save detailed results to JSON
    json_filename = 'sensitivity_budget_detailed.json'
    with open(json_filename, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("\n" + "="*70)
    print("BUDGET SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  - {output_filename} (CSV)")
    print(f"  - {json_filename} (JSON)")
    print(f"\nNext step: Run plot_sensitivity_budget.py to visualize results")
    print("="*70)

if __name__ == "__main__":
    run_budget_sensitivity_analysis()
