"""
Visualization for Bike Lane Bonus Sensitivity Analysis
Creates plots showing how metrics change with bike lane benefit
"""

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_bikebonus_sensitivity_data():
    """Load bike bonus sensitivity analysis results"""
    
    # Load CSV for easy plotting
    csv_data = {}
    with open('sensitivity_bikebonus_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['BikeLane_Bonus'] == 'ERROR':
                continue
            instance_key = f"{row['Network']}_{row['Instance']}"
            if instance_key not in csv_data:
                csv_data[instance_key] = []
            csv_data[instance_key].append(row)
    
    # Load JSON for detailed data
    with open('sensitivity_bikebonus_detailed.json', 'r') as f:
        json_data = json.load(f)
    
    return csv_data, json_data

def get_short_label(instance_name):
    """Convert instance name to short label"""
    # Handle combined format like "EasternMassachusetts_EM_CNDP_20_1"
    if 'EasternMassachusetts_EM_CNDP' in instance_name:
        parts = instance_name.split('_')
        # Extract EM_20_1 from EasternMassachusetts_EM_CNDP_20_1
        return f"{parts[1]}_{parts[3]}_{parts[4]}"
    # Handle direct instance format like "EM_CNDP_20_1"
    parts = instance_name.split('_')
    if 'CNDP' in instance_name and len(parts) >= 4:
        return f"{parts[0]}_{parts[2]}_{parts[3]}"
    return instance_name

def plot_bike_car_shares_bikebonus(data, output_dir='sensitivity_plots_bikebonus'):
    """Plot bike and car shares vs bike bonus"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        # Extract data
        bonus_vals = [float(r['BikeLane_Bonus']) * 100 for r in rows]  # Convert to percentage
        bike_shares = [float(r['Bike_Share']) for r in rows]
        car_shares = [float(r['Car_Share']) for r in rows]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(bonus_vals, bike_shares, 'b-o', label='Bike Share', linewidth=2, markersize=8)
        ax.plot(bonus_vals, car_shares, 'r-s', label='Car Share', linewidth=2, markersize=8)
        
        ax.set_xlabel('Bike Lane Bonus (%)', fontsize=12)
        ax.set_ylabel('Travel Time Share', fontsize=12)
        ax.set_title(f'Bike vs Car Share: {short_label}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 95])
        
        output_file = f"{output_dir}/{short_label}_bikebonus_bike_car_shares.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_selected_links_bikebonus(data, output_dir='sensitivity_plots_bikebonus'):
    """Plot number of selected links vs bike bonus"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v', 'p']
    for idx, (instance_key, rows) in enumerate(data.items()):
        short_label = get_short_label(instance_key)
        
        bonus_vals = [float(r['BikeLane_Bonus']) * 100 for r in rows]
        num_links = [int(r['Num_Selected_Links']) for r in rows]
        max_links = int(rows[0]['Max_Possible_Links'])
        
        # Use markers only (no lines) for discrete integer values
        ax.scatter(bonus_vals, num_links, marker=markers[idx % len(markers)], 
                   s=80, label=f'{short_label} (max={max_links})')
    
    ax.set_xlabel('Bike Lane Bonus (%)', fontsize=12)
    ax.set_ylabel('Number of Selected Bike Lanes', fontsize=12)
    ax.set_title('Selected Bike Lanes vs Bike Bonus', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 95])
    # Set y-axis to show only integers
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    output_file = f"{output_dir}/bikebonus_selected_links_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_bb_nodes_bikebonus(data, output_dir='sensitivity_plots_bikebonus'):
    """Plot number of BB nodes vs bike bonus"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        bonus_vals = [float(r['BikeLane_Bonus']) * 100 for r in rows]
        bb_nodes = [int(r['Num_BB_Nodes']) for r in rows]
        
        ax.plot(bonus_vals, bb_nodes, '-o', label=short_label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Bike Lane Bonus (%)', fontsize=12)
    ax.set_ylabel('Number of BB Nodes', fontsize=12)
    ax.set_title('Computational Effort vs Bike Bonus', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 95])
    
    output_file = f"{output_dir}/bikebonus_bb_nodes_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_objective_bikebonus(data, output_dir='sensitivity_plots_bikebonus'):
    """Plot objective value vs bike bonus"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        bonus_vals = [float(r['BikeLane_Bonus']) * 100 for r in rows]
        objectives = [float(r['Objective_Total']) for r in rows]
        
        ax.plot(bonus_vals, objectives, '-o', label=short_label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Bike Lane Bonus (%)', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Objective Value vs Bike Bonus', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 95])
    
    output_file = f"{output_dir}/bikebonus_objective_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_car_bike_ratio_bikebonus(data, output_dir='sensitivity_plots_bikebonus'):
    """Plot car/bike ratio vs bike bonus"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        bonus_vals = [float(r['BikeLane_Bonus']) * 100 for r in rows]
        ratios = [float(r['Car_to_Bike_Ratio']) for r in rows]
        
        ax.plot(bonus_vals, ratios, '-o', label=short_label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Bike Lane Bonus (%)', fontsize=12)
    ax.set_ylabel('Car/Bike Ratio', fontsize=12)
    ax.set_title('Car to Bike Travel Time Ratio vs Bike Bonus', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Equal (ratio=1)')
    ax.set_xlim([0, 95])
    
    output_file = f"{output_dir}/bikebonus_car_bike_ratio_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_benefit_effectiveness(data, output_dir='sensitivity_plots_bikebonus'):
    """Plot effectiveness: reduction in objective per % bike bonus"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        bonus_vals = np.array([float(r['BikeLane_Bonus']) * 100 for r in rows])
        objectives = np.array([float(r['Objective_Total']) for r in rows])
        
        if len(bonus_vals) > 1:
            # Calculate marginal benefit (change in objective per bonus increase)
            bonus_diffs = np.diff(bonus_vals)
            obj_diffs = np.diff(objectives)
            marginal = obj_diffs / bonus_diffs
            bonus_mid = (bonus_vals[:-1] + bonus_vals[1:]) / 2
            
            ax.plot(bonus_mid, marginal, '-o', label=short_label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Bike Lane Bonus (%)', fontsize=12)
    ax.set_ylabel('Marginal Benefit (ΔObj/ΔBonus%)', fontsize=12)
    ax.set_title('Marginal Benefit of Bike Lane Bonus', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    output_file = f"{output_dir}/bikebonus_marginal_benefit.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_combined_dashboard_bikebonus(data, output_dir='sensitivity_plots_bikebonus'):
    """Create a simplified dashboard for each instance with objective and selected links"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        # Extract data
        bonus_vals = np.array([float(r['BikeLane_Bonus']) * 100 for r in rows])
        num_links = [int(r['Num_Selected_Links']) for r in rows]
        objectives = np.array([float(r['Objective_Total']) for r in rows])
        
        # Create 1x2 subplot dashboard
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Objective Value
        ax1.plot(bonus_vals, objectives, 'orange', marker='v', linewidth=2, markersize=6)
        ax1.set_xlabel('Bike Bonus (%)', fontsize=11)
        ax1.set_ylabel('Objective Value', fontsize=11)
        ax1.set_title('Total Objective', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 95])
        
        # Plot 2: Selected Links (line plot)
        ax2.plot(bonus_vals, num_links, 'cyan', marker='o', linewidth=2, markersize=6)
        ax2.set_xlabel('Bike Bonus (%)', fontsize=11)
        ax2.set_ylabel('Selected Bike Lanes', fontsize=11)
        ax2.set_title('Number of Selected Lanes', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 95])
        ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        output_file = f"{output_dir}/{short_label}_bikebonus_dashboard.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def generate_bikebonus_summary_table(data, output_dir='sensitivity_plots_bikebonus'):
    """Generate summary table for bike bonus sensitivity"""
    Path(output_dir).mkdir(exist_ok=True)
    
    output_file = f"{output_dir}/bikebonus_sensitivity_summary.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*110 + "\n")
        f.write("BIKE BONUS SENSITIVITY ANALYSIS SUMMARY (w = 1.0, Budget = 50%)\n")
        f.write("="*110 + "\n\n")
        
        for instance_key, rows in data.items():
            short_label = get_short_label(instance_key)
            f.write(f"\n{short_label}\n")
            f.write("-"*110 + "\n")
            f.write(f"{'Bonus%':>8} {'Bike':>12} {'Car':>12} {'Ratio':>10} {'BB Nodes':>10} "
                   f"{'Links':>8} {'Objective':>12} {'Status':>12}\n")
            f.write("-"*110 + "\n")
            
            for row in rows:
                f.write(f"{float(row['BikeLane_Bonus'])*100:>8.0f} "
                       f"{float(row['Bike_Share']):>12.2f} "
                       f"{float(row['Car_Share']):>12.2f} "
                       f"{float(row['Car_to_Bike_Ratio']):>10.4f} "
                       f"{int(row['Num_BB_Nodes']):>10} "
                       f"{int(row['Num_Selected_Links']):>8} "
                       f"{float(row['Objective_Total']):>12.2f} "
                       f"{row['Status']:>12}\n")
            f.write("\n")
    
    print(f"Saved summary table: {output_file}")

def main():
    """Generate all bike bonus sensitivity analysis plots"""
    
    print("="*70)
    print("BIKE BONUS SENSITIVITY ANALYSIS VISUALIZATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    try:
        csv_data, json_data = load_bikebonus_sensitivity_data()
        print(f"Loaded data for {len(csv_data)} instances")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run SensitivityAnalysis_bikebonus.py first!")
        return
    
    # Generate plots
    print("\nGenerating plots...")
    print("-"*70)
    
    print("\n1. Bike vs Car Shares...")
    plot_bike_car_shares_bikebonus(csv_data)
    
    print("\n2. Selected Links vs Bike Bonus...")
    plot_selected_links_bikebonus(csv_data)
    
    print("\n3. BB Nodes vs Bike Bonus...")
    plot_bb_nodes_bikebonus(csv_data)
    
    print("\n4. Objective vs Bike Bonus...")
    plot_objective_bikebonus(csv_data)
    
    print("\n5. Car/Bike Ratio vs Bike Bonus...")
    plot_car_bike_ratio_bikebonus(csv_data)
    
    print("\n6. Marginal Benefit Analysis...")
    plot_benefit_effectiveness(csv_data)
    
    print("\n7. Combined Dashboards...")
    plot_combined_dashboard_bikebonus(csv_data)
    
    print("\n8. Summary Table...")
    generate_bikebonus_summary_table(csv_data)
    
    print("\n" + "="*70)
    print("All plots generated successfully!")
    print("Check 'sensitivity_plots_bikebonus/' directory for outputs")
    print("="*70)

if __name__ == "__main__":
    main()
