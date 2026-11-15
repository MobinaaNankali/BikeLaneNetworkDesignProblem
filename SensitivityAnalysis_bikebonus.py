"""
Sensitivity Analysis: Varying Bike Lane Bonus
Analyzes how the bike lane benefit (travel time reduction) affects solution
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

def calculate_bikebonus_values(n_points=11):
    """
    Calculate bike bonus values to test.
    Range from 0% to 90% in increments of 9%
    
    Returns:
        Array of bike bonus values (0.0 to 0.9)
    """
    bikebonus_values = np.linspace(0.0, 0.9, n_points)  # 0%, 9%, 18%, ..., 90%
    return bikebonus_values

def run_bikebonus_sensitivity_analysis():
    """
    Run experiments with varying bike lane bonus values
    """
    
    # Calculate bike bonus values to test
    bikebonus_values = calculate_bikebonus_values(n_points=11)
    
    print("="*70)
    print("SENSITIVITY ANALYSIS: Bike Lane Bonus")
    print("="*70)
    print(f"\nTesting {len(bikebonus_values)} different bike bonus values:")
    print(f"Bike bonus range: {bikebonus_values[0]:.2f} to {bikebonus_values[-1]:.2f}")
    print(f"Bike bonus values: {[f'{b:.2f}' for b in bikebonus_values]}")
    print("\nFixed parameters:")
    print("  - w = 1.0 (equal weight for car and bike)")
    print("  - Budget = 50% of total cost")
    print("  - No UB initialization (for consistency)")
    print("="*70)
    
    # Results storage
    results = []
    detailed_results = {}
    
    # CSV output
    output_filename = 'sensitivity_bikebonus_results.csv'
    headers = [
        'Network',
        'Instance',
        'BikeLane_Bonus',
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
        total_experiments = len(bikebonus_values) * len([ins for ins_list in SELECTED_INSTANCES.values() for ins in ins_list])
        
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
                
                # Loop through bike bonus values
                for bonus_val in bikebonus_values:
                    experiment_count += 1
                    print(f"  [{experiment_count}/{total_experiments}] Bike Bonus = {bonus_val:.2f} ({bonus_val*100:.0f}%)")
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
                        
                        # Create solver with default parameters
                        solver = NewOA_DNDP_withtimelimit.OA_DNDP_CG(
                            network,
                            inflate_cost,
                            use_ub_initialization=False,
                            budget_percentage=50.0  # Keep budget at 50%
                        )
                        
                        # Set w = 1 (equal weight)
                        solver.w = 1.0
                        
                        # Set bike lane bonus
                        solver.bikelane_bonus = bonus_val
                        
                        # Time limit for each run
                        solver.timelimit = 1800 * 8  # 30 minutes max per run
                        
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
                        obj_bike = bike_share
                        obj_car = car_share
                        car_to_bike_ratio = car_share / bike_share if bike_share > 1e-6 else float('inf')
                        
                        # Collect selected link names
                        selected_link_names = [str(link) for link in selected_links]
                        
                        # Store results
                        result_row = [
                            net,
                            ins,
                            bonus_val,
                            50.0,  # Budget percentage
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
                        bonus_key = f"bonus_{bonus_val:.2f}"
                        detailed_results[instance_key][bonus_key] = {
                            'bike_bonus': bonus_val,
                            'budget_pct': 50.0,
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
                            net, ins, bonus_val, 50.0,
                            'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR',
                            'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR', '', 'ERROR', 'ERROR'
                        ]
                        writer.writerow(error_row)
                        csvfile.flush()
    
    # Save detailed results as JSON
    json_filename = 'sensitivity_bikebonus_detailed.json'
    with open(json_filename, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("\n" + "="*70)
    print("BIKE BONUS SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_filename}")
    print(f"Detailed results saved to: {json_filename}")
    print(f"Total experiments completed: {experiment_count}")
    print("="*70)

if __name__ == "__main__":
    run_bikebonus_sensitivity_analysis()
