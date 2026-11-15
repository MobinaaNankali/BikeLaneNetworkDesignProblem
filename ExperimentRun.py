import csv
import time
from src import Network
from src import OA_DNDP_CG
'''

# --- Define all experiments here ---
# You can add/remove networks or instances from this dictionary
EXPERIMENTS11 = {
    'SiouxFalls': [
        'SF_CNDP_10_3',
        'SF_CNDP_20_3',
        'SF_CNDP_30_3'
        
    ],
    'EasternMassachusetts': [
        'EM_CNDP_10_3',
        'EM_CNDP_20_3',
        'EM_CNDP_30_3'
    ],
    
    'Anaheim': [
        'A_CNDP_10_3',
        'A_CNDP_20_3',
        'A_CNDP_30_3',
        'A_CNDP_60_3'
    ]
}

EXPERIMENTS = {
    'SiouxFalls': [
        'SF_CNDP_20_3',
        'SF_CNDP_30_1',
        'SF_CNDP_30_2',
        'SF_CNDP_30_3'
    ],
    
}

# --- Helper Dictionaries (from your main file) ---
scal_flow = {
    'SiouxFalls': 1e-3,
    'EasternMassachusetts': 1e-3,
    'BerlinMitteCenter': 1e-3,
    'BerlinMitteCenter_fixed': 1e-3,
    'Anaheim': 1e-3,
    'Barcelona': 1e-3,
    'Braess': 1,
    'HarkerFriesz': 1
}
inflate_trips = {
    'SiouxFalls': 1,
    'EasternMassachusetts': 4,
    'BerlinMitteCenter': 2,
    'BerlinMitteCenter_fixed': 2,
    'Anaheim': 4,
    'Barcelona': 2,
    'Braess': 1,
    'HarkerFriesz': 0.25
}

# --- Main execution ---
def run_all_experiments():
    output_filename = 'results.csv'
    
    # These will be the headers in the CSV file
    headers = [
        'Network', 
        'Instance', 
        'Num_Candidates', 
        'Num_BB_Nodes', 
        'Gap_Percent', 
        'Objective_UB', 
        'Bike_Share', 
        'Car_Share', 
        'Selected_Links', # ADDED
        'Selected_link_list', # ADDED
        'Total_Time_s', 
        'TAP_Time_s', 
        'CG_Time_s (Pricing)', 
        'RMP_Time_s'
    ]
    
    # Open the CSV file to write the results
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers) # Write the header row
        
        print(f"Starting experiments... Results will be saved to {output_filename}")
        
        # Loop through each network defined in EXPERIMENTS
        for net, instances in EXPERIMENTS.items():
            
            # Loop through each instance for the current network
            for ins in instances:
                print(f"\n--- RUNNING: {net} ({ins}) ---")
                
                # --- Setup constants (from your main file) ---
                scale_dem = 1
                b_prop = 0.5
                inflate_cost = 1
                
                # Try to parse the candidate number from the instance name
                try:
                    num_candidates = int(ins.split('_')[2])
                except (IndexError, ValueError):
                    num_candidates = 'N/A' # Handle names like 'HF_CNDP_1'

                # --- Initialize Network and Solver ---
                print(f"Initializing network {net}...")
                network = Network.Network(net, ins, b_prop, 1e-0, scal_flow[net], inflate_trips[net] * scale_dem)
                print(f"TD: {network.TD}, TD_bike: {network.TD_bike}")
                
                test = OA_DNDP_CG.OA_DNDP_CG(network, inflate_cost)
                
                # --- Run the solver ---
                print("Solving...")
                try:
                    # This now unpacks all the returned values
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
                        Selected_link_list # ADDED
                    ) = test.solve()
                    
                    print(f"--- COMPLETED: {net} ({ins}) ---")
                    
                    # --- Prepare the result row ---
                    result_row = [
                        net,
                        ins,
                        num_candidates,
                        num_bb_nodes,
                        gap,
                        best_ub,
                        bike_share,
                        car_share,
                        selected_links,
                        Selected_link_list,
                        tot_time,
                        tap_time,
                        pricing_time,
                        rmp_time
                    ]
                    
                    # --- Write to CSV ---
                    writer.writerow(result_row)
                    
                except Exception as e:
                    print(f"!!! FAILED: {net} ({ins}) with error: {e} !!!")
                    # Write an error row to the CSV
                    writer.writerow([net, ins, num_candidates, 'ERROR', str(e)])
                
                # Flush the file buffer to save results immediately
                csvfile.flush()

        print(f"\nAll experiments finished. Results saved to {output_filename}.")

# This makes the script runnable
if __name__ == "__main__":
    run_all_experiments()
'''

import csv
import time
from src import Network
from src import NewOA_DNDP_withtimelimit
#from src import OA_DNDP_CG_improved
#from src import NewOA_DNDP_WithUBH


def attach_tap_counter(test):
    """Count every traffic-assignment (TAP) solve without editing the solver source.

    A TAP solve is exactly one call to `network.tapas(...)`. In this solver that
    happens on two paths: the UB heuristic (evaluateSolution -> network.tapas) and
    the B&B (self.TAP -> network.tapas). Wrapping `network.tapas` catches both,
    whereas wrapping self.TAP would miss the heuristic solves. The wrapper shadows
    the bound method on the network instance (instance attributes take precedence),
    so `self.network.tapas(...)` inside the solver routes through it.
    Must be called before test.solve()."""
    test._tap_count = 0
    network = test.network
    original_tapas = network.tapas

    def counting_tapas(*args, **kwargs):
        test._tap_count += 1
        return original_tapas(*args, **kwargs)

    network.tapas = counting_tapas
    return test


def count_initial_paths(network):
    """initRMP seeds exactly one path per positive-demand OD pair (car + bike).
    Recompute that count from the network so we don't have to modify the solver."""
    initial = 0
    for r in network.origins:
        for s in network.zones:
            if r.getDemand(s) > 0:
                initial += 1
            if r.getDemandBike(s) > 0:
                initial += 1
    return initial


def collect_solver_metrics(test, network):
    """Read the extra reporting metrics off the solver object after solve()."""
    # RMP size (exact, straight from the CPLEX model): rows (constraints) and
    # columns (variables). Variables grow with column generation (path flow vars).
    try:
        rmp_constraints = test.rmp.number_of_constraints
    except Exception:
        rmp_constraints = 'N/A'
    try:
        rmp_variables = test.rmp.number_of_variables
    except Exception:
        rmp_variables = 'N/A'

    # RMP composition: value-function cut points and OA (TSTT/Beckmann) cut points.
    num_vf_cuts = len(test.vf_cuts) if hasattr(test, 'vf_cuts') else 'N/A'
    if hasattr(test, 'tstt_cuts'):
        num_oa_cuts = sum(len(cuts) for cuts in test.tstt_cuts.values())
    else:
        num_oa_cuts = 'N/A'

    # Column-generation paths: initial seed vs. generated during CG.
    try:
        total_paths = len(test.getCarPaths()) + len(test.getBikePaths())
    except Exception:
        total_paths = 'N/A'
    initial_paths = count_initial_paths(network)
    cg_paths = total_paths - initial_paths if isinstance(total_paths, int) else 'N/A'

    return {
        'rmp_constraints': rmp_constraints,
        'rmp_variables': rmp_variables,
        'num_vf_cuts': num_vf_cuts,
        'num_oa_cuts': num_oa_cuts,
        'total_paths': total_paths,
        'initial_paths': initial_paths,
        'cg_paths': cg_paths,
        'tap_solved': getattr(test, '_tap_count', 'N/A'),
    }

# --- Define all experiments here ---
# You can add/remove networks or instances from this dictionary
EXPERIMENTS1 = {
    'SiouxFalls': [
        'SF_CNDP_10_3',
        'SF_CNDP_20_3',
        'SF_CNDP_30_3'
        
    ],
    'EasternMassachusetts': [
        'EM_CNDP_10_3',
        'EM_CNDP_20_3',
        'EM_CNDP_30_3'
    ],
    
    'Anaheim': [
        'A_CNDP_10_3',
        'A_CNDP_20_3',
        'A_CNDP_30_3',
        'A_CNDP_60_3'
    ]
}

EXPERIMENTS = {
    'Anaheim': [
        'A_CNDP_10_1',
        'A_CNDP_10_2',
        'A_CNDP_10_3',
        'A_CNDP_20_1',
        'A_CNDP_20_2',
        'A_CNDP_20_3',
        'A_CNDP_30_1',
        'A_CNDP_30_2',
        'A_CNDP_30_3',
        'A_CNDP_60_1',
        'A_CNDP_60_2',
        'A_CNDP_60_3'
    ],
    
}

EXPERIMENTSEM = {
    'EasternMassachusetts': [
        'EM_CNDP_10_1',
        'EM_CNDP_10_2',
        'EM_CNDP_10_3',
        'EM_CNDP_20_1',
        'EM_CNDP_20_2',
        'EM_CNDP_20_3',
        'EM_CNDP_30_1',
        'EM_CNDP_30_2',
        'EM_CNDP_30_3'
    ]
    
}
EXPERIMENTSSS = {
    'SiouxFalls': [
        'SF_CNDP_10_1',
        'SF_CNDP_10_2',
        'SF_CNDP_10_3',
        'SF_CNDP_20_1',
        'SF_CNDP_20_2',
        'SF_CNDP_20_3',
        'SF_CNDP_30_1',
        'SF_CNDP_30_2',
        'SF_CNDP_30_3'
    ]
}
EXPERIMENTSBMCF = {
    'BerlinMitteCenter_fixed': [
        'BMC_CNDP_10_1',
        'BMC_CNDP_10_2',
        'BMC_CNDP_10_3',
        'BMC_CNDP_20_1',
        'BMC_CNDP_20_2',
        'BMC_CNDP_20_3',
        'BMC_CNDP_30_1',
        'BMC_CNDP_30_2',
        'BMC_CNDP_30_3'
    ]
}

# --- Helper Dictionaries (from your main file) ---
scal_flow = {
    'SiouxFalls': 1e-3,
    'EasternMassachusetts': 1e-3,
    'BerlinMitteCenter': 1e-3,
    'BerlinMitteCenter_fixed': 1e-3,
    'Anaheim': 1e-3,
    'Barcelona': 1e-3,
    'Braess': 1,
    'HarkerFriesz': 1
}
inflate_trips = {
    'SiouxFalls': 1,
    'EasternMassachusetts': 4,
    'BerlinMitteCenter': 2,
    'BerlinMitteCenter_fixed': 2,
    'Anaheim': 4,
    'Barcelona': 2,
    'Braess': 1,
    'HarkerFriesz': 0.25
}

# --- Main execution ---
def run_all_experiments():
    # --- Run parameters (single place to change) ---
    RUN_W = 5              # objective weight on car TSTT
    RUN_BONUS = 0.4       # tau: bike-lane travel-time reduction factor
    RUN_TIMELIMIT = 3600 *4   # per-instance time limit in seconds (1 hour)

    # Self-describing filename so each run is identifiable and never overwrites a
    # previous one: which network(s), w, the bike-lane bonus (tau), a "diag" tag
    # marking the extended diagnostic columns (RMP size, cut counts, path counts,
    # TAP count), and a timestamp.
    run_stamp = time.strftime('%Y%m%d_%H%M%S')
    nets_tag = '-'.join(EXPERIMENTS.keys())
    output_filename = f'results_{nets_tag}_diag_w{RUN_W}_tau{RUN_BONUS}_{run_stamp}.csv'

    # These will be the headers in the CSV file
    headers = [
        'Network',
        'Instance',
        'Num_Candidates',
        'Status',  # 'Converged', 'TimeLimit', or 'TreeExhausted'
        'Num_BB_Nodes',
        'Gap_Percent',
        'Objective_UB',
        'Bike_Share',
        'Car_Share',
        'Selected_Links',
        'Selected_link_list',
        'Total_Time_s',
        'TAP_Time_s',
        'CG_Time_s (Pricing)',
        'RMP_Time_s',
        # --- Additional termination diagnostics ---
        'RMP_Total_Constraints',
        'RMP_Total_Variables',
        'Num_VF_Cuts',
        'Num_OA_Cuts',
        'Num_CG_Paths',
        'Num_Initial_Paths',
        'Num_Total_Paths',
        'Num_TAP_Solved'
    ]
    
    # Open the CSV file to write the results
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        print(f"Starting experiments... Results will be saved to {output_filename}")
        
        # Flag to track if we've written parameters yet
        params_written = False
        
        # Loop through each network defined in EXPERIMENTS
        for net, instances in EXPERIMENTS.items():
            
            # Loop through each instance for the current network
            for ins in instances:
                print(f"\n--- RUNNING: {net} ({ins}) ---")
                
                # --- Setup constants (from your main file) ---
                scale_dem = 1
                b_prop = 0.5
                inflate_cost = 1
                
                # Try to parse the candidate number from the instance name
                try:
                    num_candidates = int(ins.split('_')[2])
                except (IndexError, ValueError):
                    num_candidates = 'N/A' # Handle names like 'HF_CNDP_1'

                # --- Initialize Network and Solver ---
                print(f"Initializing network {net}...")
                network = Network.Network(net, ins, b_prop, 1e-0, scal_flow[net], inflate_trips[net] * scale_dem)
                print(f"TD: {network.TD}, TD_bike: {network.TD_bike}")
                
                # Create solver with w=1 and budget=50%
                test = NewOA_DNDP_withtimelimit.OA_DNDP_CG(
                    network, 
                    inflate_cost, 
                    use_ub_initialization=False, 
                    budget_percentage=50.0
                )
                test.w = RUN_W                    # objective weight on car TSTT
                test.bikelane_bonus = RUN_BONUS   # tau: bike-lane travel-time reduction
                test.timelimit = RUN_TIMELIMIT    # per-instance time limit (seconds)

                # Intercept TAP calls to count them (must be before solve()).
                attach_tap_counter(test)

                # Write parameter information on first run only (uses actual values from test object)
                if not params_written:
                    writer.writerow(['# Parameters'])
                    writer.writerow(['w', 'bike_bonus', 'budget_percentage'])
                    writer.writerow([test.w, test.bikelane_bonus, test.budget_percentage])
                    writer.writerow([])  # Empty row for separation
                    writer.writerow(headers) # Write the header row
                    params_written = True  
                
                # --- Run the solver ---
                print("Solving...")
                try:
                    # solve() returns 12 values on TimeLimit/Converged (with ub_history)
                    # but only 11 on TreeExhausted (no ub_history). Unpack tolerantly so
                    # the best solution found is always saved, never lost to a ValueError.
                    result = test.solve()
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
                        Selected_link_list,
                    ) = result[:11]
                    ub_history = result[11] if len(result) > 11 else []
                    
                    # Determine status based on time and gap
                    if tot_time >= test.timelimit:
                        status = 'TimeLimit'
                    elif gap < test.params.min_NDP_gap:
                        status = 'Converged'
                    else:
                        status = 'TreeExhausted'
                    
                    print(f"--- COMPLETED: {net} ({ins}) - Status: {status} ---")

                    # --- Gather extra termination diagnostics off the solver object ---
                    metrics = collect_solver_metrics(test, network)
                    print(
                        f"    RMP size: constraints={metrics['rmp_constraints']} variables={metrics['rmp_variables']}, "
                        f"VF cuts={metrics['num_vf_cuts']}, OA cuts={metrics['num_oa_cuts']}, "
                        f"paths init/CG/total={metrics['initial_paths']}/{metrics['cg_paths']}/{metrics['total_paths']}, "
                        f"TAP solved={metrics['tap_solved']}"
                    )

                    # --- Prepare the result row ---
                    result_row = [
                        net,
                        ins,
                        num_candidates,
                        status,
                        num_bb_nodes,
                        gap,
                        best_ub,
                        bike_share,
                        car_share,
                        selected_links_str,  # String representation
                        ';'.join([str(link) for link in Selected_link_list]),  # Convert list to string
                        tot_time,
                        tap_time,
                        pricing_time,
                        rmp_time,
                        metrics['rmp_constraints'],
                        metrics['rmp_variables'],
                        metrics['num_vf_cuts'],
                        metrics['num_oa_cuts'],
                        metrics['cg_paths'],
                        metrics['initial_paths'],
                        metrics['total_paths'],
                        metrics['tap_solved']
                    ]
                    
                    # --- Write to CSV ---
                    writer.writerow(result_row)
                    
                except Exception as e:
                    print(f"!!! FAILED: {net} ({ins}) with error: {e} !!!")
                    # Write an error row to the CSV
                    writer.writerow([net, ins, num_candidates, 'ERROR', str(e)])
                
                # Flush the file buffer to save results immediately
                csvfile.flush()

        print(f"\nAll experiments finished. Results saved to {output_filename}.")

# This makes the script runnable
if __name__ == "__main__":
    run_all_experiments()