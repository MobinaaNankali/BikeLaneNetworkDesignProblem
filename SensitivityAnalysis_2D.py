"""
2D Sensitivity Analysis: Varying Both w and Budget
Analyzes the interaction between w (car vs bike priority) and budget
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

def calculate_parameter_grid(n_w=7, n_budget=7):
    """
    Calculate parameter grid for 2D sensitivity analysis.
    
    Args:
        n_w: Number of w values to test
        n_budget: Number of budget percentages to test
    
    Returns:
        w_values: Array of w values (log-spaced)
        budget_percentages: Array of budget percentages (linear-spaced)
    """
    # w values: logarithmically spaced from 0.1 to 10
    w_values = np.logspace(np.log10(0.1), np.log10(10), n_w)
    
    # Budget percentages: linearly spaced from 20% to 80%
    budget_percentages = np.linspace(20, 80, n_budget)
    
    return w_values, budget_percentages

def run_2d_sensitivity_analysis():
    """
    Run experiments varying both w and budget percentage
    """
    
    # Calculate parameter grid
    w_values, budget_percentages = calculate_parameter_grid(n_w=7, n_budget=7)
    
    total_experiments = len(w_values) * len(budget_percentages) * len([ins for ins_list in SELECTED_INSTANCES.values() for ins in ins_list])
    
    print("="*70)
    print("2D SENSITIVITY ANALYSIS: w × Budget")
    print("="*70)
    print(f"\nParameter Grid:")
    print(f"  w values ({len(w_values)}): {[f'{w:.2f}' for w in w_values]}")
    print(f"  Budget percentages ({len(budget_percentages)}): {[f'{b:.0f}%' for b in budget_percentages]}")
    print(f"\nTotal experiments per instance: {len(w_values) * len(budget_percentages)}")
    print(f"Total experiments: {total_experiments}")
    print(f"\nNo UB initialization (for consistency)")
    print("="*70)
    
    # Results storage
    results = []
    detailed_results = {}
    
    # CSV output
    output_filename = 'sensitivity_2d_results.csv'
    headers = [
        'Network',
        'Instance',
        'w_value',
        'Budget_Percent',
        'Budget_Value',
        'Total_Cost',
        'Num_BB_Nodes',
        'Gap_Percent',
        'Objective_Total',
        'Objective_Bike',
        'Objective_Car',
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
        
        experiment_count = 0
        
        # Loop through selected instances
        for net, instances in SELECTED_INSTANCES.items():
            for ins in instances:
                print(f"\n{'='*70}")
                print(f"Instance: {net} - {ins}")
                print(f"{'='*70}")
                
                instance_key = f"{net}_{ins}"
                detailed_results[instance_key] = {}
                
                # First, calculate total cost for this instance
                network_temp = Network.Network(
                    net, ins, 0.5, 1e-0,
                    scal_flow[net],
                    inflate_trips[net]
                )
                solver_temp = NewOA_DNDP_withtimelimit.OA_DNDP_CG(
                    network_temp, 1, use_ub_initialization=False, budget_percentage=50.0
                )
                total_cost = solver_temp.budget * 2
                num_candidates = len(solver_temp.varlinks)
                
                print(f"Total possible cost: {total_cost:.2f}")
                print(f"Number of candidate links: {num_candidates}")
                print(f"{'='*70}\n")
                
                # Loop through parameter grid
                for w_val in w_values:
                    for budget_pct in budget_percentages:
                        experiment_count += 1
                        print(f"  [{experiment_count}/{total_experiments}] w={w_val:.3f}, Budget={budget_pct:.0f}%")
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
                            
                            # Create solver with specified parameters
                            solver = NewOA_DNDP_withtimelimit.OA_DNDP_CG(
                                network,
                                inflate_cost,
                                use_ub_initialization=False,
                                budget_percentage=budget_pct
                            )
                            
                            # Set w value
                            solver.w = w_val
                            
                            # Shorter time limit for 2D analysis
                            solver.timelimit = 1200  # 20 minutes max per run
                            
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
                            
                            # Calculate metrics
                            # obj_bike and obj_car are already in bike_share and car_share
                            obj_bike = bike_share
                            obj_car = car_share
                            car_to_bike_ratio = car_share / bike_share if bike_share > 1e-6 else float('inf')
                            
                            # Collect selected link names
                            selected_link_names = [str(link) for link in selected_links]
                            
                            # Store results
                            result_row = [
                                net,
                                ins,
                                w_val,
                                budget_pct,
                                solver.budget,
                                total_cost,
                                num_bb_nodes,
                                gap,
                                best_ub,
                                obj_bike,
                                obj_car,
                                bike_share,
                                car_share,
                                car_to_bike_ratio,
                                len(selected_links),
                                num_candidates,
                                ';'.join(selected_link_names),
                                tot_time,
                                'OPTIMAL' if gap < 1e-3 else 'TIME_LIMIT'
                            ]
                            
                            writer.writerow(result_row)
                            csvfile.flush()
                            
                            # Store detailed results
                            param_key = f"w_{w_val:.3f}_budget_{budget_pct:.0f}"
                            detailed_results[instance_key][param_key] = {
                                'w': w_val,
                                'budget_pct': budget_pct,
                                'budget_value': solver.budget,
                                'objective': best_ub,
                                'obj_bike': obj_bike,
                                'obj_car': obj_car,
                                'bike_share': bike_share,
                                'car_share': car_share,
                                'num_bb_nodes': num_bb_nodes,
                                'gap': gap,
                                'selected_links': selected_link_names,
                                'total_time': tot_time
                            }
                            
                            print(f"  ✓ Objective: {best_ub:.2f} | Bike: {bike_share:.2f} | Car: {car_share:.2f} | Links: {len(selected_links)}/{num_candidates}")
                            
                        except Exception as e:
                            print(f"  ✗ ERROR: {str(e)}")
                            error_row = [
                                net, ins, w_val, budget_pct,
                                'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR',
                                'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR', '', 'ERROR', 'ERROR'
                            ]
                            writer.writerow(error_row)
                            csvfile.flush()
    
    # Save detailed results as JSON
    json_filename = 'sensitivity_2d_detailed.json'
    with open(json_filename, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("\n" + "="*70)
    print("2D SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_filename}")
    print(f"Detailed results saved to: {json_filename}")
    print(f"Total experiments completed: {experiment_count}")
    print("="*70)

if __name__ == "__main__":
    run_2d_sensitivity_analysis()
