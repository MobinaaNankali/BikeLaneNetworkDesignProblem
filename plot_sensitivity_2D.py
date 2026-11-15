"""
2D Contour Plot Visualization for w × Budget Sensitivity Analysis
Creates contour plots showing how metrics change across parameter space
"""

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import cm
from matplotlib.ticker import LogLocator, FuncFormatter

def load_2d_sensitivity_data():
    """Load 2D sensitivity analysis results"""
    
    # Load CSV for plotting
    csv_data = {}
    with open('sensitivity_2d_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['w_value'] == 'ERROR':
                continue
            instance_key = f"{row['Network']}_{row['Instance']}"
            if instance_key not in csv_data:
                csv_data[instance_key] = []
            csv_data[instance_key].append(row)
    
    # Load JSON for detailed data
    with open('sensitivity_2d_detailed.json', 'r') as f:
        json_data = json.load(f)
    
    return csv_data, json_data

def get_short_label(instance_name):
    """Convert instance name to short label"""
    parts = instance_name.split('_')
    if 'CNDP' in instance_name and len(parts) >= 4:
        return f"{parts[0]}_{parts[2]}_{parts[3]}"
    return instance_name

def prepare_grid_data(rows, metric_key):
    """
    Prepare data for contour plotting.
    
    Args:
        rows: List of result dictionaries
        metric_key: Key for the metric to plot
    
    Returns:
        w_grid, budget_grid, metric_grid
    """
    # Extract unique values
    w_values = sorted(list(set(float(r['w_value']) for r in rows)))
    budget_values = sorted(list(set(float(r['Budget_Percent']) for r in rows)))
    
    # Create grids
    n_w = len(w_values)
    n_budget = len(budget_values)
    
    metric_grid = np.zeros((n_budget, n_w))
    
    # Fill the grid
    for row in rows:
        w = float(row['w_value'])
        budget = float(row['Budget_Percent'])
        
        try:
            metric_value = float(row[metric_key])
        except (ValueError, KeyError):
            metric_value = np.nan
        
        i = budget_values.index(budget)
        j = w_values.index(w)
        metric_grid[i, j] = metric_value
    
    # Create meshgrids for plotting
    w_grid, budget_grid = np.meshgrid(w_values, budget_values)
    
    return w_grid, budget_grid, metric_grid, w_values, budget_values

def plot_contour_bike_share(data, output_dir='sensitivity_plots_2d'):
    """Create contour plot for bike share"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        w_grid, budget_grid, metric_grid, w_vals, budget_vals = prepare_grid_data(rows, 'Bike_Share')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create contour plot
        contour = ax.contourf(w_grid, budget_grid, metric_grid, levels=15, cmap='viridis')
        contour_lines = ax.contour(w_grid, budget_grid, metric_grid, levels=10, colors='white', 
                                   linewidths=0.5, alpha=0.4)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Bike Travel Time Share', fontsize=11)
        
        # Labels
        ax.set_xlabel('w (Car Weight)', fontsize=12)
        ax.set_ylabel('Budget (%)', fontsize=12)
        
        # Log scale for w axis
        ax.set_xscale('log')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        output_file = f"{output_dir}/{short_label}_contour_bike_share.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_contour_car_share(data, output_dir='sensitivity_plots_2d'):
    """Create contour plot for car share"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        w_grid, budget_grid, metric_grid, w_vals, budget_vals = prepare_grid_data(rows, 'Car_Share')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(w_grid, budget_grid, metric_grid, levels=15, cmap='Reds')
        contour_lines = ax.contour(w_grid, budget_grid, metric_grid, levels=10, colors='white', 
                                   linewidths=0.5, alpha=0.4)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Car Travel Time Share', fontsize=11)
        
        ax.set_xlabel('w (Car Weight)', fontsize=12)
        ax.set_ylabel('Budget (%)', fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        output_file = f"{output_dir}/{short_label}_contour_car_share.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_contour_car_bike_ratio(data, output_dir='sensitivity_plots_2d'):
    """Create contour plot for car/bike ratio"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        w_grid, budget_grid, metric_grid, w_vals, budget_vals = prepare_grid_data(rows, 'Car_to_Bike_Ratio')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(w_grid, budget_grid, metric_grid, levels=15, cmap='RdYlGn_r')
        contour_lines = ax.contour(w_grid, budget_grid, metric_grid, levels=10, colors='black', 
                                   linewidths=0.5, alpha=0.3)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        
        # Add special contour line for ratio = 1 (equilibrium)
        contour_eq = ax.contour(w_grid, budget_grid, metric_grid, levels=[1.0], 
                                colors='blue', linewidths=2, linestyles='--')
        ax.clabel(contour_eq, inline=True, fontsize=10, fmt='Equilibrium (1.0)')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Car/Bike Ratio', fontsize=11)
        
        ax.set_xlabel('w (Car Weight)', fontsize=12)
        ax.set_ylabel('Budget (%)', fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        output_file = f"{output_dir}/{short_label}_contour_ratio.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_contour_selected_links(data, output_dir='sensitivity_plots_2d'):
    """Create contour plot for number of selected links"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        w_grid, budget_grid, metric_grid, w_vals, budget_vals = prepare_grid_data(rows, 'Num_Selected_Links')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(w_grid, budget_grid, metric_grid, levels=15, cmap='plasma')
        contour_lines = ax.contour(w_grid, budget_grid, metric_grid, levels=10, colors='white', 
                                   linewidths=0.5, alpha=0.4)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Number of Selected Bike Lanes', fontsize=11)
        
        ax.set_xlabel('w (Car Weight)', fontsize=12)
        ax.set_ylabel('Budget (%)', fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        output_file = f"{output_dir}/{short_label}_contour_selected_links.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_contour_bb_nodes(data, output_dir='sensitivity_plots_2d'):
    """Create contour plot for number of BB nodes (computational effort)"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        w_grid, budget_grid, metric_grid, w_vals, budget_vals = prepare_grid_data(rows, 'Num_BB_Nodes')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(w_grid, budget_grid, metric_grid, levels=15, cmap='coolwarm')
        contour_lines = ax.contour(w_grid, budget_grid, metric_grid, levels=10, colors='black', 
                                   linewidths=0.5, alpha=0.3)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Number of BB Nodes', fontsize=11)
        
        ax.set_xlabel('w (Car Weight)', fontsize=12)
        ax.set_ylabel('Budget (%)', fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        output_file = f"{output_dir}/{short_label}_contour_bb_nodes.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_contour_objective(data, output_dir='sensitivity_plots_2d'):
    """Create contour plot for objective value"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        w_grid, budget_grid, metric_grid, w_vals, budget_vals = prepare_grid_data(rows, 'Objective_Total')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(w_grid, budget_grid, metric_grid, levels=15, cmap='magma')
        contour_lines = ax.contour(w_grid, budget_grid, metric_grid, levels=10, colors='white', 
                                   linewidths=0.5, alpha=0.4)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Objective Value', fontsize=11)
        
        ax.set_xlabel('w (Car Weight)', fontsize=12)
        ax.set_ylabel('Budget (%)', fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        output_file = f"{output_dir}/{short_label}_contour_objective.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_combined_contour_dashboard(data, output_dir='sensitivity_plots_2d'):
    """Create a comprehensive 2x3 dashboard of contour plots"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for instance_key, rows in data.items():
        short_label = get_short_label(instance_key)
        
        # Prepare data for all metrics
        w_grid_bike, budget_grid_bike, bike_grid, _, _ = prepare_grid_data(rows, 'Bike_Share')
        w_grid_car, budget_grid_car, car_grid, _, _ = prepare_grid_data(rows, 'Car_Share')
        w_grid_ratio, budget_grid_ratio, ratio_grid, _, _ = prepare_grid_data(rows, 'Car_to_Bike_Ratio')
        w_grid_links, budget_grid_links, links_grid, _, _ = prepare_grid_data(rows, 'Num_Selected_Links')
        w_grid_bb, budget_grid_bb, bb_grid, _, _ = prepare_grid_data(rows, 'Num_BB_Nodes')
        w_grid_obj, budget_grid_obj, obj_grid, _, _ = prepare_grid_data(rows, 'Objective_Total')
        
        # Create dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Plot 1: Bike Share
        ax = axes[0, 0]
        cs1 = ax.contourf(w_grid_bike, budget_grid_bike, bike_grid, levels=15, cmap='viridis')
        ax.contour(w_grid_bike, budget_grid_bike, bike_grid, levels=10, colors='white', linewidths=0.5, alpha=0.3)
        ax.set_xlabel('w (Car Weight)', fontsize=10)
        ax.set_ylabel('Budget (%)', fontsize=10)
        ax.set_title('Bike Share', fontweight='bold', fontsize=11)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(cs1, ax=ax, label='Bike Share')
        
        # Plot 2: Car Share
        ax = axes[0, 1]
        cs2 = ax.contourf(w_grid_car, budget_grid_car, car_grid, levels=15, cmap='Reds')
        ax.contour(w_grid_car, budget_grid_car, car_grid, levels=10, colors='white', linewidths=0.5, alpha=0.3)
        ax.set_xlabel('w (Car Weight)', fontsize=10)
        ax.set_ylabel('Budget (%)', fontsize=10)
        ax.set_title('Car Share', fontweight='bold', fontsize=11)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(cs2, ax=ax, label='Car Share')
        
        # Plot 3: Car/Bike Ratio
        ax = axes[0, 2]
        cs3 = ax.contourf(w_grid_ratio, budget_grid_ratio, ratio_grid, levels=15, cmap='RdYlGn_r')
        ax.contour(w_grid_ratio, budget_grid_ratio, ratio_grid, levels=[1.0], 
                  colors='blue', linewidths=2, linestyles='--')
        ax.set_xlabel('w (Car Weight)', fontsize=10)
        ax.set_ylabel('Budget (%)', fontsize=10)
        ax.set_title('Car/Bike Ratio', fontweight='bold', fontsize=11)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(cs3, ax=ax, label='Ratio')
        
        # Plot 4: Selected Links
        ax = axes[1, 0]
        cs4 = ax.contourf(w_grid_links, budget_grid_links, links_grid, levels=15, cmap='plasma')
        ax.contour(w_grid_links, budget_grid_links, links_grid, levels=10, colors='white', linewidths=0.5, alpha=0.3)
        ax.set_xlabel('w (Car Weight)', fontsize=10)
        ax.set_ylabel('Budget (%)', fontsize=10)
        ax.set_title('Selected Bike Lanes', fontweight='bold', fontsize=11)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(cs4, ax=ax, label='# Links')
        
        # Plot 5: BB Nodes
        ax = axes[1, 1]
        cs5 = ax.contourf(w_grid_bb, budget_grid_bb, bb_grid, levels=15, cmap='coolwarm')
        ax.contour(w_grid_bb, budget_grid_bb, bb_grid, levels=10, colors='black', linewidths=0.5, alpha=0.3)
        ax.set_xlabel('w (Car Weight)', fontsize=10)
        ax.set_ylabel('Budget (%)', fontsize=10)
        ax.set_title('BB Nodes', fontweight='bold', fontsize=11)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(cs5, ax=ax, label='# BB Nodes')
        
        # Plot 6: Objective
        ax = axes[1, 2]
        cs6 = ax.contourf(w_grid_obj, budget_grid_obj, obj_grid, levels=15, cmap='magma')
        ax.contour(w_grid_obj, budget_grid_obj, obj_grid, levels=10, colors='white', linewidths=0.5, alpha=0.3)
        ax.set_xlabel('w (Car Weight)', fontsize=10)
        ax.set_ylabel('Budget (%)', fontsize=10)
        ax.set_title('Objective Value', fontweight='bold', fontsize=11)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(cs6, ax=ax, label='Objective')
        
        output_file = f"{output_dir}/{short_label}_contour_dashboard.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def main():
    """Generate all 2D sensitivity contour plots"""
    
    print("="*70)
    print("2D SENSITIVITY ANALYSIS VISUALIZATION (w × Budget)")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    try:
        csv_data, json_data = load_2d_sensitivity_data()
        print(f"Loaded data for {len(csv_data)} instances")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run SensitivityAnalysis_2D.py first!")
        return
    
    # Generate contour plots
    print("\nGenerating contour plots...")
    print("-"*70)
    
    print("\n1. Bike Share Contours...")
    plot_contour_bike_share(csv_data)
    
    print("\n2. Car Share Contours...")
    plot_contour_car_share(csv_data)
    
    print("\n3. Car/Bike Ratio Contours...")
    plot_contour_car_bike_ratio(csv_data)
    
    print("\n4. Selected Links Contours...")
    plot_contour_selected_links(csv_data)
    
    print("\n5. BB Nodes Contours...")
    plot_contour_bb_nodes(csv_data)
    
    print("\n6. Objective Value Contours...")
    plot_contour_objective(csv_data)
    
    print("\n7. Combined Dashboards...")
    plot_combined_contour_dashboard(csv_data)
    
    print("\n" + "="*70)
    print("All contour plots generated successfully!")
    print("Check 'sensitivity_plots_2d/' directory for outputs")
    print("="*70)

if __name__ == "__main__":
    main()
