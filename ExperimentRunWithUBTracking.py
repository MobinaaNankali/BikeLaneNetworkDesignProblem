import csv
import time
import json
from src import Network
from src import NewOA_DNDP_withtimelimit

# --- Define all experiments here ---
EXPERIMENTS = {
    'EasternMassachusetts': [
        'EM_CNDP_10_1',
    ]
}

# --- Helper Dictionaries ---
scal_flow = {
    'SiouxFalls': 1e-3,
    'EasternMassachusetts': 1e-3,
    'BerlinMitteCenter': 1e-3,
    'Anaheim': 1e-3,
    'Barcelona': 1e-3,
    'Braess': 1,
    'HarkerFriesz': 1
}
inflate_trips = {
    'SiouxFalls': 1,
    'EasternMassachusetts': 4,
    'BerlinMitteCenter': 2,
    'Anaheim': 4,
    'Barcelona': 2,
    'Braess': 1,
    'HarkerFriesz': 0.25
}

# --- Main execution ---
def run_experiments_with_ub_tracking():
    """
    Run experiments twice for each instance:
    1. Without UB initialization
    2. With UB initialization
    
    Track and save UB values at each BB node iteration.
    """
    
    # CSV file for main results
    results_filename = 'results_ub_comparison.csv'
    
    # JSON file for UB tracking data (easier for plotting)
    ub_tracking_filename = 'ub_tracking_data.json'
    
    # Dictionary to store all UB tracking data
    ub_tracking_data = {}
    
    # Headers for results CSV
    headers = [
        'Network', 
        'Instance', 
        'UB_Init',  # 'Yes' or 'No'
        'Num_Candidates', 
        'Status',
        'Num_BB_Nodes', 
        'Gap_Percent', 
        'Objective_UB', 
        'Bike_Share', 
        'Car_Share', 
        'Selected_Links',
        'Selected_link_list',
        'Total_Time_s', 
        'TAP_Time_s', 
        'CG_Time_s', 
        'RMP_Time_s'
    ]
    
    # Open the CSV file to write the results
    with open(results_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        print(f"Starting experiments with UB tracking... Results will be saved to {results_filename}")
        print(f"UB tracking data will be saved to {ub_tracking_filename}")
        
        # Loop through each network
        for net, instances in EXPERIMENTS.items():
            
            # Loop through each instance
            for ins in instances:
                
                # Run TWICE: once without UB init, once with UB init
                for use_ub_init in [False, True]:
                    
                    ub_init_label = "With_UB_Init" if use_ub_init else "Without_UB_Init"
                    print(f"\n{'='*60}")
                    print(f"RUNNING: {net} ({ins}) - {ub_init_label}")
                    print(f"{'='*60}")
                    
                    # --- Setup constants ---
                    scale_dem = 1
                    b_prop = 0.5
                    inflate_cost = 1
                    
                    # Parse candidate number
                    try:
                        num_candidates = int(ins.split('_')[2])
                    except (IndexError, ValueError):
                        num_candidates = 'N/A'

                    # --- Initialize Network and Solver ---
                    print(f"Initializing network {net}...")
                    network = Network.Network(
                        net, ins, b_prop, 1e-0, 
                        scal_flow[net], 
                        inflate_trips[net] * scale_dem
                    )
                    print(f"TD: {network.TD}, TD_bike: {network.TD_bike}")
                    
                    # Create solver with or without UB initialization
                    test = NewOA_DNDP_withtimelimit.OA_DNDP_CG(
                        network, 
                        inflate_cost, 
                        use_ub_initialization=use_ub_init
                    )
                    
                    # --- Run the solver ---
                    print("Solving...")
                    try:
                        # Unpack all returned values including ub_history
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
                            ub_history  # List of (iteration, ub) tuples
                        ) = test.solve()
                        
                        # Determine status
                        if tot_time >= test.timelimit:
                            status = 'TimeLimit'
                        elif gap < test.params.min_NDP_gap:
                            status = 'Converged'
                        else:
                            status = 'TreeExhausted'
                        
                        print(f"\n--- COMPLETED: {net} ({ins}) - {ub_init_label} ---")
                        print(f"Status: {status}, BB Nodes: {num_bb_nodes}, Gap: {gap:.6g}")
                        
                        # --- Store UB tracking data ---
                        tracking_key = f"{net}_{ins}_{ub_init_label}"
                        ub_tracking_data[tracking_key] = {
                            'network': net,
                            'instance': ins,
                            'ub_init': use_ub_init,
                            'ub_init_label': ub_init_label,
                            'ub_history': ub_history,  # List of (iteration, ub) tuples
                            'final_ub': best_ub,
                            'num_bb_nodes': num_bb_nodes,
                            'status': status
                        }
                        
                        # --- Write to results CSV ---
                        result_row = [
                            net,
                            ins,
                            'Yes' if use_ub_init else 'No',
                            num_candidates,
                            status,
                            num_bb_nodes,
                            gap,
                            best_ub,
                            bike_share,
                            car_share,
                            selected_links,
                            selected_link_list,
                            tot_time,
                            tap_time,
                            pricing_time,
                            rmp_time
                        ]
                        
                        writer.writerow(result_row)
                        csvfile.flush()
                        
                    except Exception as e:
                        print(f"!!! FAILED: {net} ({ins}) - {ub_init_label} with error: {e} !!!")
                        import traceback
                        traceback.print_exc()
                        writer.writerow([net, ins, 'Yes' if use_ub_init else 'No', num_candidates, 'ERROR', str(e)])
                        csvfile.flush()
        
        print(f"\nAll experiments finished. Results saved to {results_filename}")
    
    # --- Save UB tracking data to JSON ---
    print(f"\nSaving UB tracking data to {ub_tracking_filename}...")
    with open(ub_tracking_filename, 'w') as f:
        json.dump(ub_tracking_data, f, indent=2)
    
    print(f"UB tracking data saved successfully!")
    print(f"\nYou can now run the plotting script to visualize the results.")

# This makes the script runnable
if __name__ == "__main__":
    run_experiments_with_ub_tracking()
