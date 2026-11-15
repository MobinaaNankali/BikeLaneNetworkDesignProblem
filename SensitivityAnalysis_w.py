"""
Sensitivity Analysis: Varying parameter w
Analyzes how the bike/car share balance changes with different w values
"""

import csv
import json
import numpy as np
from src import Network
from src import NewOA_DNDP_withtimelimit

# Configuration
SELECTED_INSTANCES = {
    'EasternMassachusetts': [
        'EM_CNDP_20_2'
    ]
}

scal_flow = {'EasternMassachusetts': 1e-3}
inflate_trips = {'EasternMassachusetts': 4}

def calculate_w_values(n_points=11):
    """
    Calculate w values to achieve different car/bike ratios.
    
    With objective: (w * car_tstt) + bike_tstt
    - Small w: bike dominates (car contribution is small)
    - Large w: car dominates (car contribution is large)
    
    We'll vary w from 0.1 to 10 to get a range of car/bike ratios
    """
    # Logarithmic spacing to cover wide range
    w_values = np.logspace(-1, 1, n_points)  # 0.1 to 10
    return w_values

def run_sensitivity_analysis():
    """
    Run experiments with varying w values
    """
    
    # Calculate w values to test
    w_values = calculate_w_values(n_points=11)  # 11 points from w=0.1 to w=10
    
    print("="*70)
    print("SENSITIVITY ANALYSIS: Parameter w")
    print("="*70)
    print(f"\nTesting {len(w_values)} different w values:")
    print(f"w range: {w_values[0]:.2f} to {w_values[-1]:.2f}")
    print("\nObjective function: (w * car_tstt) + bike_tstt")
    print("  - Small w → bike travel time dominates")
    print("  - Large w → car travel time dominates")
    print("="*70)
    
    # Results storage
    results = []
    detailed_results = {}
    
    # CSV output
    output_filename = 'sensitivity_w_results.csv'
    headers = [
        'Network',
        'Instance',
        'w_value',
        'Num_BB_Nodes',
        'Gap_Percent',
        'Objective_Total',
        'Bike_Share',
        'Car_Share',
        'Car_to_Bike_Ratio',
        'Num_Selected_Links',
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
                
                # Loop through w values
                for w_val in w_values:
                    print(f"\n  Testing w = {w_val:.4f}")
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
                        
                        # Create solver WITHOUT UB initialization
                        solver = NewOA_DNDP_withtimelimit.OA_DNDP_CG(
                            network,
                            inflate_cost,
                            use_ub_initialization=False
                        )
                        
                        # Set the w parameter
                        solver.w = w_val
                        
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
                        print(f"    Selected Links: {num_selected}")
                        print(f"    Time: {tot_time:.2f}s")
                        
                        # Store results
                        result_row = [
                            net,
                            ins,
                            w_val,
                            num_bb_nodes,
                            gap,
                            best_ub,
                            bike_share,
                            car_share,
                            car_to_bike_ratio,
                            num_selected,
                            ';'.join(selected_link_list),
                            tot_time,
                            status
                        ]
                        
                        writer.writerow(result_row)
                        csvfile.flush()
                        
                        # Store for JSON
                        instance_results[f"w_{w_val:.4f}"] = {
                            'w': w_val,
                            'num_bb_nodes': num_bb_nodes,
                            'gap': gap,
                            'objective': best_ub,
                            'bike_share': bike_share,
                            'car_share': car_share,
                            'car_to_bike_ratio': car_to_bike_ratio,
                            'selected_links': selected_link_list,
                            'num_selected': num_selected,
                            'total_time': tot_time,
                            'status': status,
                            'ub_history': ub_history
                        }
                        
                    except Exception as e:
                        print(f"    ERROR: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        
                        writer.writerow([net, ins, w_val, 'ERROR', str(e)])
                        csvfile.flush()
                
                # Store instance results
                detailed_results[f"{net}_{ins}"] = instance_results
    
    # Save detailed results to JSON
    json_filename = 'sensitivity_w_detailed.json'
    with open(json_filename, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  - {output_filename} (CSV)")
    print(f"  - {json_filename} (JSON)")
    print(f"\nNext step: Run plot_sensitivity_w.py to visualize results")
    print("="*70)

if __name__ == "__main__":
    run_sensitivity_analysis()
