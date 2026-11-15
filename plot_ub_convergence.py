"""
Plotting script for UB convergence comparison
Visualizes upper bound evolution over BB node iterations
Comparing scenarios with and without UB initialization
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_ub_tracking_data(filename='ub_tracking_data.json'):
    """Load UB tracking data from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def get_short_label(instance_name):
    """Convert instance name to short label (e.g., EM_CNDP_10_1 -> EM_10_1)"""
    # Remove network prefix if present
    parts = instance_name.split('_')
    # For names like "EM_CNDP_10_1", extract "EM_10_1"
    if 'CNDP' in instance_name:
        if len(parts) >= 4:
            return f"{parts[0]}_{parts[2]}_{parts[3]}"
    # For other formats, try to shorten
    return instance_name

def plot_ub_convergence_per_instance(data, output_dir='plots'):
    """
    Create individual plots for each instance comparing with/without UB init
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Group data by network and instance
    instances = {}
    for key, value in data.items():
        instance_key = f"{value['network']}_{value['instance']}"
        if instance_key not in instances:
            instances[instance_key] = {}
        
        ub_init_status = 'with_ub_init' if value['ub_init'] else 'without_ub_init'
        instances[instance_key][ub_init_status] = value
    
    # Create a plot for each instance
    for instance_key, instance_data in instances.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot without UB initialization (if available)
        if 'without_ub_init' in instance_data:
            without_data = instance_data['without_ub_init']
            iterations, ubs = zip(*without_data['ub_history'])
            ax.plot(iterations, ubs, 'b-o', label='Without Init', 
                   linewidth=2, markersize=6, alpha=0.7)
        
        # Plot with UB initialization (if available)
        if 'with_ub_init' in instance_data:
            with_data = instance_data['with_ub_init']
            iterations, ubs = zip(*with_data['ub_history'])
            ax.plot(iterations, ubs, 'r-s', label='With Init', 
                   linewidth=2, markersize=6, alpha=0.7)
        
        # Use short label for title
        short_label = get_short_label(instance_data[list(instance_data.keys())[0]]['instance'])
        
        ax.set_xlabel('BB Node Iteration', fontsize=12)
        ax.set_ylabel('Upper Bound', fontsize=12)
        #ax.set_title(f'UB Convergence: {short_label}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save the figure
        output_file = f"{output_dir}/{short_label}_convergence.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_all_instances_combined(data, output_dir='plots'):
    """
    Create a combined plot showing all instances
    Separate subplots for with/without UB initialization
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Group data
    instances = {}
    for key, value in data.items():
        instance_key = f"{value['network']}_{value['instance']}"
        if instance_key not in instances:
            instances[instance_key] = {}
        
        ub_init_status = 'with_ub_init' if value['ub_init'] else 'without_ub_init'
        instances[instance_key][ub_init_status] = value
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color map for different instances
    colors = plt.cm.tab10(np.linspace(0, 1, len(instances)))
    
    # Plot all instances without UB init
    for idx, (instance_key, instance_data) in enumerate(instances.items()):
        if 'without_ub_init' in instance_data:
            without_data = instance_data['without_ub_init']
            short_label = get_short_label(without_data['instance'])
            
            iterations, ubs = zip(*without_data['ub_history'])
            ax1.plot(iterations, ubs, '-o', label=short_label, 
                    color=colors[idx], linewidth=2, markersize=4, alpha=0.7)
    
    ax1.set_xlabel('BB Node Iteration', fontsize=12)
    ax1.set_ylabel('Upper Bound', fontsize=12)
    ax1.set_title('Without UB Initialization', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot all instances with UB init
    for idx, (instance_key, instance_data) in enumerate(instances.items()):
        if 'with_ub_init' in instance_data:
            with_data = instance_data['with_ub_init']
            short_label = get_short_label(with_data['instance'])
            
            iterations, ubs = zip(*with_data['ub_history'])
            ax2.plot(iterations, ubs, '-s', label=short_label, 
                    color=colors[idx], linewidth=2, markersize=4, alpha=0.7)
    
    ax2.set_xlabel('BB Node Iteration', fontsize=12)
    ax2.set_ylabel('Upper Bound', fontsize=12)
    ax2.set_title('With UB Initialization', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('UB Convergence Comparison Across All Instances', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    output_file = f"{output_dir}/all_instances_combined.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_overlay_comparison(data, output_dir='plots'):
    """
    Create overlay plots where each instance shows both scenarios on same plot
    All instances in one figure with subplots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Group data
    instances = {}
    for key, value in data.items():
        instance_key = f"{value['network']}_{value['instance']}"
        if instance_key not in instances:
            instances[instance_key] = {}
        
        ub_init_status = 'with_ub_init' if value['ub_init'] else 'without_ub_init'
        instances[instance_key][ub_init_status] = value
    
    # Determine grid size
    n_instances = len(instances)
    n_cols = 3
    n_rows = (n_instances + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    
    # Flatten axes array for easier indexing
    if n_instances > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each instance
    for idx, (instance_key, instance_data) in enumerate(instances.items()):
        ax = axes[idx]
        
        # Get short label
        short_label = get_short_label(instance_data[list(instance_data.keys())[0]]['instance'])
        
        # Plot without UB init
        if 'without_ub_init' in instance_data:
            without_data = instance_data['without_ub_init']
            iterations, ubs = zip(*without_data['ub_history'])
            ax.plot(iterations, ubs, 'b-o', label='Without Init', 
                   linewidth=2, markersize=5, alpha=0.7)
        
        # Plot with UB init
        if 'with_ub_init' in instance_data:
            with_data = instance_data['with_ub_init']
            iterations, ubs = zip(*with_data['ub_history'])
            ax.plot(iterations, ubs, 'r-s', label='With Init', 
                   linewidth=2, markersize=5, alpha=0.7)
        
        ax.set_xlabel('BB Node Iteration', fontsize=10)
        ax.set_ylabel('Upper Bound', fontsize=10)
        ax.set_title(short_label, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_instances, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('UB Convergence: Individual Instance Comparisons', 
                 fontsize=16, fontweight='bold')
    
    output_file = f"{output_dir}/overlay_comparison_grid.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_percentage_improvement(data, output_dir='plots'):
    """
    Plot the percentage improvement in UB when using initialization
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Group data
    instances = {}
    for key, value in data.items():
        instance_key = f"{value['network']}_{value['instance']}"
        if instance_key not in instances:
            instances[instance_key] = {}
        
        ub_init_status = 'with_ub_init' if value['ub_init'] else 'without_ub_init'
        instances[instance_key][ub_init_status] = value
    
    improvements = {}
    for instance_key, instance_data in instances.items():
        if 'with_ub_init' in instance_data and 'without_ub_init' in instance_data:
            with_ub = instance_data['with_ub_init']['final_ub']
            without_ub = instance_data['without_ub_init']['final_ub']
            
            # Calculate percentage improvement
            improvement = ((without_ub - with_ub) / without_ub) * 100
            short_label = get_short_label(instance_data['with_ub_init']['instance'])
            improvements[short_label] = improvement
    
    # Create bar plot
    if improvements:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        instances_list = list(improvements.keys())
        values = list(improvements.values())
        
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax.bar(range(len(instances_list)), values, color=colors, alpha=0.7)
        ax.set_xticks(range(len(instances_list)))
        ax.set_xticklabels(instances_list, rotation=45, ha='right')
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title('UB Improvement with Initialization (% reduction)', 
                    fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}%',
                   ha='center', va='bottom' if value > 0 else 'top', fontsize=9)
        
        output_file = f"{output_dir}/ub_improvement_comparison.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def generate_summary_report(data, output_file='plots/summary_report.txt'):
    """Generate a text summary of the results"""
    Path(output_file).parent.mkdir(exist_ok=True)
    
    # Group data
    instances = {}
    for key, value in data.items():
        instance_key = f"{value['network']}_{value['instance']}"
        if instance_key not in instances:
            instances[instance_key] = {}
        
        ub_init_status = 'with_ub_init' if value['ub_init'] else 'without_ub_init'
        instances[instance_key][ub_init_status] = value
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("UB INITIALIZATION COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for instance_key, instance_data in instances.items():
            short_label = get_short_label(instance_data[list(instance_data.keys())[0]]['instance'])
            f.write(f"\n{short_label}\n")
            f.write("-"*80 + "\n")
            
            if 'without_ub_init' in instance_data:
                without = instance_data['without_ub_init']
                f.write(f"  WITHOUT UB Init:\n")
                f.write(f"    Final UB: {without['final_ub']:.6f}\n")
                f.write(f"    BB Nodes: {without['num_bb_nodes']}\n")
                f.write(f"    Status: {without['status']}\n")
            
            if 'with_ub_init' in instance_data:
                with_init = instance_data['with_ub_init']
                f.write(f"  WITH UB Init:\n")
                f.write(f"    Final UB: {with_init['final_ub']:.6f}\n")
                f.write(f"    BB Nodes: {with_init['num_bb_nodes']}\n")
                f.write(f"    Status: {with_init['status']}\n")
            
            if 'with_ub_init' in instance_data and 'without_ub_init' in instance_data:
                improvement = ((without['final_ub'] - with_init['final_ub']) / 
                             without['final_ub']) * 100
                node_reduction = without['num_bb_nodes'] - with_init['num_bb_nodes']
                f.write(f"  IMPROVEMENT:\n")
                f.write(f"    UB Reduction: {improvement:.2f}%\n")
                f.write(f"    BB Node Reduction: {node_reduction}\n")
    
    print(f"Saved summary report: {output_file}")

def main():
    """Main function to generate all plots"""
    print("="*60)
    print("UB Convergence Plotting Script")
    print("="*60)
    
    # Load data
    print("\nLoading UB tracking data...")
    try:
        data = load_ub_tracking_data('ub_tracking_data.json')
        print(f"Loaded data for {len(data)} experiment runs")
    except FileNotFoundError:
        print("ERROR: ub_tracking_data.json not found!")
        print("Please run ExperimentRunWithUBTracking.py first.")
        return
    
    # Create plots
    print("\nGenerating plots...")
    print("-"*60)
    
    print("\n1. Creating individual instance comparison plots...")
    plot_ub_convergence_per_instance(data)
    
    print("\n2. Creating combined comparison plot...")
    plot_all_instances_combined(data)
    
    print("\n3. Creating overlay comparison grid...")
    plot_overlay_comparison(data)
    
    print("\n4. Creating improvement comparison plot...")
    plot_percentage_improvement(data)
    
    print("\n5. Generating summary report...")
    generate_summary_report(data)
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("Check the 'plots' directory for output files.")
    print("="*60)

if __name__ == "__main__":
    main()
