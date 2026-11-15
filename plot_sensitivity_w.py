"""
Visualization for Sensitivity Analysis: Parameter w
Creates various plots showing how metrics change with w
"""

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def load_sensitivity_data():
    """Load sensitivity analysis results"""
    
    # Load CSV for easy plotting
    csv_data = {}
    with open('sensitivity_w_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['w_value'] == 'ERROR':
                continue
            instance_key = f"{row['Network']}_{row['Instance']}"
            if instance_key not in csv_data:
                csv_data[instance_key] = []
            csv_data[instance_key].append(row)
    
    # Load JSON for detailed data
    with open('sensitivity_w_detailed.json', 'r') as f:
        json_data = json.load(f)
    
    return csv_data, json_data

def get_short_label(instance_name):
    """Convert instance name to short label"""
    parts = instance_name.split('_')
    if 'CNDP' in instance_name and len(parts) >= 4:
        return f"{parts[0]}_{parts[2]}_{parts[3]}"
    return instance_name

def plot_bike_car_shares(data, output_dir='sensitivity_plots'):
    """Plot bike and car shares vs w"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        # Extract data
        w_vals = [float(r['w_value']) for r in rows]
        bike_shares = [float(r['Bike_Share']) for r in rows]
        car_shares = [float(r['Car_Share']) for r in rows]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(w_vals, bike_shares, 'b-o', label='Bike Share', linewidth=2, markersize=8)
        ax.plot(w_vals, car_shares, 'r-s', label='Car Share', linewidth=2, markersize=8)
        
        ax.set_xlabel('Weight Parameter (w)', fontsize=12)
        ax.set_ylabel('Travel Time Share', fontsize=12)
        ax.set_title(f'Bike vs Car Share: {short_label}', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_file = f"{output_dir}/{short_label}_bike_car_shares.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_car_bike_ratio(data, output_dir='sensitivity_plots'):
    """Plot car/bike ratio vs w"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        w_vals = [float(r['w_value']) for r in rows]
        ratios = [float(r['Car_to_Bike_Ratio']) for r in rows]
        
        ax.plot(w_vals, ratios, '-o', label=short_label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Weight Parameter (w)', fontsize=12)
    ax.set_ylabel('Car/Bike Ratio', fontsize=12)
    ax.set_title('Car to Bike Travel Time Ratio vs w', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Equal (ratio=1)')
    
    output_file = f"{output_dir}/car_bike_ratio_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_total_time(data, output_dir='sensitivity_plots'):
    """Plot total computation time vs w"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        w_vals = [float(r['w_value']) for r in rows]
        total_time = [float(r['Total_Time_s']) for r in rows]
        
        ax.plot(w_vals, total_time, '-o', label=short_label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Weight Parameter (w)', fontsize=12)
    ax.set_ylabel('Total Time (seconds)', fontsize=12)
    ax.set_title('Computational Effort vs w', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    output_file = f"{output_dir}/total_time_vs_w.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_selected_links(data, output_dir='sensitivity_plots'):
    """Plot number of selected links vs w"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        w_vals = [float(r['w_value']) for r in rows]
        num_links = [int(r['Num_Selected_Links']) for r in rows]
        
        ax.plot(w_vals, num_links, '-o', label=short_label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Weight Parameter (w)', fontsize=12)
    ax.set_ylabel('Number of Selected Bike Lanes', fontsize=12)
    ax.set_title('Selected Bike Lanes vs w', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    output_file = f"{output_dir}/selected_links_vs_w.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_objective_components(data, output_dir='sensitivity_plots'):
    """Plot objective components vs w"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        w_vals = [float(r['w_value']) for r in rows]
        bike_shares = [float(r['Bike_Share']) for r in rows]
        car_shares = [float(r['Car_Share']) for r in rows]
        objectives = [float(r['Objective_Total']) for r in rows]
        
        # Calculate weighted car component
        weighted_car = [w * c for w, c in zip(w_vals, car_shares)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(w_vals, bike_shares, 'b-o', label='Bike Component', linewidth=2, markersize=6)
        ax.plot(w_vals, weighted_car, 'r-s', label='Weighted Car Component (w × car)', 
                linewidth=2, markersize=6)
        ax.plot(w_vals, objectives, 'g-^', label='Total Objective', 
                linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Weight Parameter (w)', fontsize=12)
        ax.set_ylabel('Objective Value', fontsize=12)
        ax.set_title(f'Objective Components: {short_label}', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_file = f"{output_dir}/{short_label}_objective_components.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_combined_dashboard(data, output_dir='sensitivity_plots'):
    """Create a comprehensive dashboard for each instance"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        # Extract data
        w_vals = [float(r['w_value']) for r in rows]
        bike_shares = [float(r['Bike_Share']) for r in rows]
        car_shares = [float(r['Car_Share']) for r in rows]
        num_links = [int(r['Num_Selected_Links']) for r in rows]
        
        # Create 1x2 subplot dashboard
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Bike vs Car Shares
        ax1.plot(w_vals, bike_shares, 'b-o', label='Bike', linewidth=2, markersize=6)
        ax1.plot(w_vals, car_shares, 'r-s', label='Car', linewidth=2, markersize=6)
        ax1.set_xlabel('w', fontsize=10)
        ax1.set_ylabel('Travel Time Share', fontsize=10)
        ax1.set_title('Bike vs Car Shares', fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Selected Links
        ax2.plot(w_vals, num_links, 'c-p', linewidth=2, markersize=6)
        ax2.set_xlabel('w', fontsize=10)
        ax2.set_ylabel('Selected Bike Lanes', fontsize=10)
        ax2.set_title('Number of Selected Lanes', fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        #plt.suptitle(f'Sensitivity Analysis Dashboard: {short_label}', 
                     #fontsize=16, fontweight='bold')
        
        output_file = f"{output_dir}/{short_label}_dashboard.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def generate_summary_table(data, output_dir='sensitivity_plots'):
    """Generate summary table"""
    Path(output_dir).mkdir(exist_ok=True)
    
    output_file = f"{output_dir}/sensitivity_summary.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("SENSITIVITY ANALYSIS SUMMARY: Parameter w\n")
        f.write("="*100 + "\n\n")
        
        for instance_key, rows in data.items():
            short_label = get_short_label(instance_key)
            f.write(f"\n{short_label}\n")
            f.write("-"*100 + "\n")
            f.write(f"{'w':>8} {'Bike':>12} {'Car':>12} {'Ratio':>10} {'BB Nodes':>10} "
                   f"{'Links':>8} {'Objective':>12} {'Status':>12}\n")
            f.write("-"*100 + "\n")
            
            for row in rows:
                f.write(f"{float(row['w_value']):>8.4f} "
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
    """Generate all sensitivity analysis plots"""
    
    print("="*70)
    print("SENSITIVITY ANALYSIS VISUALIZATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    try:
        csv_data, json_data = load_sensitivity_data()
        print(f"Loaded data for {len(csv_data)} instances")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run SensitivityAnalysis_w.py first!")
        return
    
    # Generate plots
    print("\nGenerating plots...")
    print("-"*70)
    
    print("\n1. Bike vs Car Shares...")
    plot_bike_car_shares(csv_data)
    
    print("\n2. Car/Bike Ratio...")
    plot_car_bike_ratio(csv_data)
    
    print("\n3. Total Time vs w...")
    plot_total_time(csv_data)
    
    print("\n4. Selected Links vs w...")
    plot_selected_links(csv_data)
    
    print("\n5. Objective Components...")
    plot_objective_components(csv_data)
    
    print("\n6. Combined Dashboards...")
    plot_combined_dashboard(csv_data)
    
    print("\n7. Summary Table...")
    generate_summary_table(csv_data)
    
    print("\n" + "="*70)
    print("All plots generated successfully!")
    print("Check 'sensitivity_plots/' directory for outputs")
    print("="*70)

if __name__ == "__main__":
    main()
