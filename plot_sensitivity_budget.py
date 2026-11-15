"""
Visualization for Budget Sensitivity Analysis
Creates various plots showing how metrics change with budget
"""

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_budget_sensitivity_data():
    """Load budget sensitivity analysis results"""
    
    # Load CSV for easy plotting
    csv_data = {}
    with open('sensitivity_budget_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Budget_Percent'] == 'ERROR':
                continue
            instance_key = f"{row['Network']}_{row['Instance']}"
            if instance_key not in csv_data:
                csv_data[instance_key] = []
            csv_data[instance_key].append(row)
    
    # Load JSON for detailed data
    with open('sensitivity_budget_detailed.json', 'r') as f:
        json_data = json.load(f)
    
    return csv_data, json_data

def get_short_label(instance_name):
    """Convert instance name to short label"""
    parts = instance_name.split('_')
    if 'CNDP' in instance_name and len(parts) >= 4:
        return f"{parts[0]}_{parts[2]}_{parts[3]}"
    return instance_name

def plot_bike_car_shares_budget(data, output_dir='sensitivity_plots'):
    """Plot bike and car shares vs budget"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        # Extract data
        budget_pcts = [float(r['Budget_Percent']) for r in rows]
        bike_shares = [float(r['Bike_Share']) for r in rows]
        car_shares = [float(r['Car_Share']) for r in rows]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(budget_pcts, bike_shares, 'b-o', label='Bike Share', linewidth=2, markersize=8)
        ax.plot(budget_pcts, car_shares, 'r-s', label='Car Share', linewidth=2, markersize=8)
        
        ax.set_xlabel('Budget (%)', fontsize=12)
        ax.set_ylabel('Travel Time Share', fontsize=12)
        ax.set_title(f'Bike vs Car Share: {short_label}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 100])
        
        output_file = f"{output_dir}/{short_label}_budget_bike_car_shares.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_selected_links_budget(data, output_dir='sensitivity_plots'):
    """Plot number of selected links vs budget"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        budget_pcts = [float(r['Budget_Percent']) for r in rows]
        num_links = [int(r['Num_Selected_Links']) for r in rows]
        max_links = int(rows[0]['Max_Possible_Links'])
        
        ax.plot(budget_pcts, num_links, '-o', label=f'{short_label} (max={max_links})', 
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Budget (%)', fontsize=12)
    ax.set_ylabel('Number of Selected Bike Lanes', fontsize=12)
    ax.set_title('Selected Bike Lanes vs Budget', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    
    output_file = f"{output_dir}/budget_selected_links_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_bb_nodes_budget(data, output_dir='sensitivity_plots'):
    """Plot number of BB nodes vs budget"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        budget_pcts = [float(r['Budget_Percent']) for r in rows]
        bb_nodes = [int(r['Num_BB_Nodes']) for r in rows]
        
        ax.plot(budget_pcts, bb_nodes, '-o', label=short_label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Budget (%)', fontsize=12)
    ax.set_ylabel('Number of BB Nodes', fontsize=12)
    ax.set_title('Computational Effort vs Budget', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    
    output_file = f"{output_dir}/budget_bb_nodes_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_objective_budget(data, output_dir='sensitivity_plots'):
    """Plot objective value vs budget"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        budget_pcts = [float(r['Budget_Percent']) for r in rows]
        objectives = [float(r['Objective_Total']) for r in rows]
        
        ax.plot(budget_pcts, objectives, '-o', label=short_label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Budget (%)', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Objective Value vs Budget', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    
    output_file = f"{output_dir}/budget_objective_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_car_bike_ratio_budget(data, output_dir='sensitivity_plots'):
    """Plot car/bike ratio vs budget"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        budget_pcts = [float(r['Budget_Percent']) for r in rows]
        ratios = [float(r['Car_to_Bike_Ratio']) for r in rows]
        
        ax.plot(budget_pcts, ratios, '-o', label=short_label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Budget (%)', fontsize=12)
    ax.set_ylabel('Car/Bike Ratio', fontsize=12)
    ax.set_title('Car to Bike Travel Time Ratio vs Budget', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Equal (ratio=1)')
    ax.set_xlim([0, 100])
    
    output_file = f"{output_dir}/budget_car_bike_ratio_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_utilization_rate(data, output_dir='sensitivity_plots'):
    """Plot budget utilization (selected links / max possible)"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        budget_pcts = [float(r['Budget_Percent']) for r in rows]
        num_selected = [int(r['Num_Selected_Links']) for r in rows]
        max_possible = int(rows[0]['Max_Possible_Links'])
        
        utilization = [(n / max_possible) * 100 for n in num_selected]
        
        ax.plot(budget_pcts, utilization, '-o', label=short_label, linewidth=2, markersize=6)
    
    # Add diagonal line showing perfect utilization
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect Utilization')
    
    ax.set_xlabel('Budget (%)', fontsize=12)
    ax.set_ylabel('Link Utilization (%)', fontsize=12)
    ax.set_title('Budget Utilization Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    
    output_file = f"{output_dir}/budget_utilization_rate.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_combined_dashboard_budget(data, output_dir='sensitivity_plots'):
    """Create a comprehensive dashboard for each instance"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        # Extract data
        budget_pcts = [float(r['Budget_Percent']) for r in rows]
        bike_shares = [float(r['Bike_Share']) for r in rows]
        car_shares = [float(r['Car_Share']) for r in rows]
        bb_nodes = [int(r['Num_BB_Nodes']) for r in rows]
        num_links = [int(r['Num_Selected_Links']) for r in rows]
        objectives = [float(r['Objective_Total']) for r in rows]
        ratios = [float(r['Car_to_Bike_Ratio']) for r in rows]
        
        # Create 2x3 subplot dashboard
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot 1: Bike vs Car Shares
        ax1.plot(budget_pcts, bike_shares, 'b-o', label='Bike', linewidth=2, markersize=5)
        ax1.plot(budget_pcts, car_shares, 'r-s', label='Car', linewidth=2, markersize=5)
        ax1.set_xlabel('Budget (%)', fontsize=10)
        ax1.set_ylabel('Travel Time Share', fontsize=10)
        ax1.set_title('Bike vs Car Shares', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 100])
        
        # Plot 2: Car/Bike Ratio
        ax2.plot(budget_pcts, ratios, 'g-^', linewidth=2, markersize=5)
        ax2.set_xlabel('Budget (%)', fontsize=10)
        ax2.set_ylabel('Car/Bike Ratio', fontsize=10)
        ax2.set_title('Car to Bike Ratio', fontweight='bold')
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 100])
        
        # Plot 3: BB Nodes
        ax3.plot(budget_pcts, bb_nodes, 'm-D', linewidth=2, markersize=5)
        ax3.set_xlabel('Budget (%)', fontsize=10)
        ax3.set_ylabel('BB Nodes', fontsize=10)
        ax3.set_title('Computational Effort', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 100])
        
        # Plot 4: Selected Links
        ax4.plot(budget_pcts, num_links, 'c-p', linewidth=2, markersize=5)
        ax4.set_xlabel('Budget (%)', fontsize=10)
        ax4.set_ylabel('Selected Bike Lanes', fontsize=10)
        ax4.set_title('Number of Selected Lanes', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 100])
        
        # Plot 5: Objective Value
        ax5.plot(budget_pcts, objectives, 'orange', marker='v', linewidth=2, markersize=5)
        ax5.set_xlabel('Budget (%)', fontsize=10)
        ax5.set_ylabel('Objective Value', fontsize=10)
        ax5.set_title('Total Objective', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([0, 100])
        
        # Plot 6: Marginal Benefit (change in objective per budget increase)
        if len(budget_pcts) > 1:
            budget_diffs = np.diff(budget_pcts)
            obj_diffs = np.diff(objectives)
            marginal = obj_diffs / budget_diffs
            budget_mid = [(budget_pcts[i] + budget_pcts[i+1])/2 for i in range(len(budget_pcts)-1)]
            ax6.plot(budget_mid, marginal, 'purple', marker='*', linewidth=2, markersize=5)
            ax6.set_xlabel('Budget (%)', fontsize=10)
            ax6.set_ylabel('Marginal Benefit', fontsize=10)
            ax6.set_title('Marginal Benefit (ΔObj/ΔBudget)', fontweight='bold')
            ax6.grid(True, alpha=0.3)
            ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'Budget Sensitivity Dashboard: {short_label}', 
                     fontsize=16, fontweight='bold')
        
        output_file = f"{output_dir}/{short_label}_budget_dashboard.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def generate_budget_summary_table(data, output_dir='sensitivity_plots'):
    """Generate summary table for budget sensitivity"""
    Path(output_dir).mkdir(exist_ok=True)
    
    output_file = f"{output_dir}/budget_sensitivity_summary.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*110 + "\n")
        f.write("BUDGET SENSITIVITY ANALYSIS SUMMARY (w = 1.0)\n")
        f.write("="*110 + "\n\n")
        
        for instance_key, rows in data.items():
            short_label = get_short_label(instance_key)
            f.write(f"\n{short_label}\n")
            f.write("-"*110 + "\n")
            f.write(f"{'Budget%':>8} {'Bike':>12} {'Car':>12} {'Ratio':>10} {'BB Nodes':>10} "
                   f"{'Links':>8} {'Objective':>12} {'Status':>12}\n")
            f.write("-"*110 + "\n")
            
            for row in rows:
                f.write(f"{float(row['Budget_Percent']):>8.0f} "
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
    """Generate all budget sensitivity analysis plots"""
    
    print("="*70)
    print("BUDGET SENSITIVITY ANALYSIS VISUALIZATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    try:
        csv_data, json_data = load_budget_sensitivity_data()
        print(f"Loaded data for {len(csv_data)} instances")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run SensitivityAnalysis_budget.py first!")
        return
    
    # Generate plots
    print("\nGenerating plots...")
    print("-"*70)
    
    print("\n1. Bike vs Car Shares...")
    plot_bike_car_shares_budget(csv_data)
    
    print("\n2. Selected Links vs Budget...")
    plot_selected_links_budget(csv_data)
    
    print("\n3. BB Nodes vs Budget...")
    plot_bb_nodes_budget(csv_data)
    
    print("\n4. Objective vs Budget...")
    plot_objective_budget(csv_data)
    
    print("\n5. Car/Bike Ratio vs Budget...")
    plot_car_bike_ratio_budget(csv_data)
    
    print("\n6. Budget Utilization Rate...")
    plot_utilization_rate(csv_data)
    
    print("\n7. Combined Dashboards...")
    plot_combined_dashboard_budget(csv_data)
    
    print("\n8. Summary Table...")
    generate_budget_summary_table(csv_data)
    
    print("\n" + "="*70)
    print("All plots generated successfully!")
    print("Check 'sensitivity_plots/' directory for outputs")
    print("="*70)

if __name__ == "__main__":
    main()
