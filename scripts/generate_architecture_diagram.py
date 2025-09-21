#!/usr/bin/env python3
"""
Simple script to generate MentorEval architecture diagram as a single image.
Saves the image directly to webpage/assets/ directory.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path

def fix_overlapping_nodes(components, min_distance=0.2):
    """Automatically adjust node positions to prevent overlapping."""
    import copy
    components = copy.deepcopy(components)
    
    # Group nodes by y-level (same row)
    y_levels = {}
    for name, props in components.items():
        y = props['pos'][1]
        if y not in y_levels:
            y_levels[y] = []
        y_levels[y].append(name)
    
    # Fix overlapping within each level
    for y, node_names in y_levels.items():
        if len(node_names) <= 1:
            continue
            
        # Sort nodes by x position
        node_names.sort(key=lambda n: components[n]['pos'][0])
        
        # Check and fix overlaps
        for i in range(len(node_names) - 1):
            node1 = node_names[i]
            node2 = node_names[i + 1]
            
            x1, y1 = components[node1]['pos']
            w1, h1 = components[node1]['size']
            x2, y2 = components[node2]['pos']
            w2, h2 = components[node2]['size']
            
            # Calculate minimum distance needed
            min_x_distance = (w1 + w2) / 2 + min_distance
            
            # If too close, move node2 to the right
            if x2 - x1 < min_x_distance:
                new_x2 = x1 + min_x_distance
                components[node2]['pos'] = (new_x2, y2)
                # Update the sorted list for next iteration
                node_names[i + 1] = node2
    
    return components

def create_architecture_diagram():
    """Create the MentorEval architecture diagram using matplotlib."""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(26, 18))
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(0, 17)
    ax.axis('off')
    
    # Define colors
    colors = {
        'lighteval': '#e1f5fe',
        'custom': '#f3e5f5', 
        'dataset': '#e8f5e8',
        'metric': '#fff3e0',
        'border_lighteval': '#01579b',
        'border_custom': '#4a148c',
        'border_dataset': '#1b5e20',
        'border_metric': '#e65100'
    }
    
    # Define more visible legend colors
    legend_colors = {
        'lighteval': '#2196F3',  # Bright blue
        'custom': '#9C27B0',     # Bright purple
        'dataset': '#4CAF50',    # Bright green
        'metric': '#FF9800'      # Bright orange
    }
    
    # Define component positions and properties
    components = {
        # Core LightEval Components - Top level
        'pipeline': {'pos': (8, 16), 'size': (2, 0.8), 'color': 'lighteval', 'text': 'LightEval\nPipeline'},
        
        # Second level - Core components
        'task_config': {'pos': (2, 14), 'size': (1.8, 0.7), 'color': 'lighteval', 'text': 'LightevalTask\nConfig'},
        'pipeline_params': {'pos': (4.5, 14), 'size': (1.8, 0.7), 'color': 'lighteval', 'text': 'Pipeline\nParameters'},
        'eval_tracker': {'pos': (7, 14), 'size': (1.8, 0.7), 'color': 'lighteval', 'text': 'Evaluation\nTracker'},
        'model_config': {'pos': (11.5, 14), 'size': (1.8, 0.7), 'color': 'lighteval', 'text': 'Model\nConfig'},
        
        # Model Configuration - Third level
        'transformers_config': {'pos': (10.5, 12), 'size': (1.5, 0.6), 'color': 'lighteval', 'text': 'Transformers\nModelConfig'},
        'litellm_config': {'pos': (12.5, 12), 'size': (1.5, 0.6), 'color': 'lighteval', 'text': 'LiteLLM\nModelConfig'},
        
        # Task Configuration - Third level
        'prompt_function': {'pos': (1, 12), 'size': (1.5, 0.6), 'color': 'lighteval', 'text': 'Prompts'},
        'metrics': {'pos': (3, 12), 'size': (1.5, 0.6), 'color': 'lighteval', 'text': 'Metrics'},
        'dataset': {'pos': (5, 12), 'size': (1.5, 0.6), 'color': 'lighteval', 'text': 'Dataset'},
        
        # Fourth level - Educational extensions
        'educational_prompts': {'pos': (0.5, 10), 'size': (1.3, 0.5), 'color': 'custom', 'text': 'Educational\nPrompts'},
        'educational_datasets': {'pos': (8.5, 10), 'size': (1.3, 0.5), 'color': 'dataset', 'text': 'Educational\nDatasets'},
        
        # Metrics System - Fourth level
        'sample_metrics': {'pos': (2.5, 10), 'size': (1.3, 0.5), 'color': 'metric', 'text': 'Sample Level\nMetrics'},
        'corpus_metrics': {'pos': (4.5, 10), 'size': (1.3, 0.5), 'color': 'metric', 'text': 'Corpus Level\nMetrics'},
        
        # Data Flow - Fourth level
        'model_response': {'pos': (10.5, 10), 'size': (1.3, 0.5), 'color': 'lighteval', 'text': 'Model\nResponse'},
        'grade_parser': {'pos': (12, 10), 'size': (1.3, 0.5), 'color': 'custom', 'text': 'Grade\nParser'},
        
        # Prompt features - Fifth level (below Educational Prompts) - PURPLE
        'guidance': {'pos': (0.2, 8), 'size': (1, 0.4), 'color': 'custom', 'text': 'Guidance'},
        'explanation': {'pos': (0.8, 8), 'size': (1, 0.4), 'color': 'custom', 'text': 'Explanation'},
        'fewshot': {'pos': (1.4, 8), 'size': (1, 0.4), 'color': 'custom', 'text': 'Few-Shot'},
        'isced': {'pos': (2, 8), 'size': (1, 0.4), 'color': 'custom', 'text': 'ISCED'},
        
        # Dataset names - Fifth level (below Educational Datasets)
        'asap': {'pos': (7.5, 8), 'size': (1, 0.4), 'color': 'dataset', 'text': 'ASAP'},
        'asap2': {'pos': (8.5, 8), 'size': (1, 0.4), 'color': 'dataset', 'text': 'ASAP2'},
        'ellipse': {'pos': (9.5, 8), 'size': (1, 0.4), 'color': 'dataset', 'text': 'ELLIPSE'},
        'mohler': {'pos': (10.5, 8), 'size': (1, 0.4), 'color': 'dataset', 'text': 'Mohler'},
        'ptasag': {'pos': (11.5, 8), 'size': (1, 0.4), 'color': 'dataset', 'text': 'PT-ASAG'},
        'arasag': {'pos': (12.5, 8), 'size': (1, 0.4), 'color': 'dataset', 'text': 'AR-ASAG'},
        
        # Sample Level Metrics - Sixth level (below prompt features and datasets)
        'exact_match': {'pos': (1.5, 6), 'size': (1, 0.4), 'color': 'metric', 'text': 'Exact\nMatch'},
        'mae': {'pos': (2.5, 6), 'size': (1, 0.4), 'color': 'metric', 'text': 'MAE'},
        'rmse': {'pos': (3.5, 6), 'size': (1, 0.4), 'color': 'metric', 'text': 'RMSE'},
        
        # Corpus Level Metrics - Sixth level (same level as sample metrics)
        'pearson': {'pos': (4.5, 6), 'size': (1, 0.4), 'color': 'metric', 'text': 'Pearson\nCorrelation'},
        'spearman': {'pos': (5.5, 6), 'size': (1, 0.4), 'color': 'metric', 'text': 'Spearman\nCorrelation'},
        'ks_stat': {'pos': (6.5, 6), 'size': (1, 0.4), 'color': 'metric', 'text': 'KS\nStatistic'},
        'wasserstein': {'pos': (7.5, 6), 'size': (1, 0.4), 'color': 'metric', 'text': 'Wasserstein\nDistance'},
    }
    
    # Automatically fix overlapping nodes
    components = fix_overlapping_nodes(components, min_distance=0.3)
    
    # Draw connections FIRST (behind nodes)
    connections = [
        # Core pipeline connections - vertical
        ('pipeline', 'task_config'),
        ('pipeline', 'pipeline_params'),
        ('pipeline', 'eval_tracker'),
        ('pipeline', 'model_config'),
        
        # Model config connections - vertical
        ('model_config', 'transformers_config'),
        ('model_config', 'litellm_config'),
        
        # Task config connections - vertical
        ('task_config', 'prompt_function'),
        ('task_config', 'metrics'),
        ('task_config', 'dataset'),
        
        # Metrics connections - vertical
        ('metrics', 'sample_metrics'),
        ('metrics', 'corpus_metrics'),
        
        # Sample level metrics - vertical
        ('sample_metrics', 'exact_match'),
        ('sample_metrics', 'mae'),
        ('sample_metrics', 'rmse'),
        
        # Corpus level metrics - vertical
        ('corpus_metrics', 'pearson'),
        ('corpus_metrics', 'spearman'),
        ('corpus_metrics', 'ks_stat'),
        ('corpus_metrics', 'wasserstein'),
        
        # Educational extensions connections
        ('prompt_function', 'educational_prompts'),
        ('dataset', 'educational_datasets'),
        
        # Prompt feature connections
        ('educational_prompts', 'guidance'),
        ('educational_prompts', 'explanation'),
        ('educational_prompts', 'fewshot'),
        ('educational_prompts', 'isced'),
        
        # Dataset connections
        ('educational_datasets', 'asap'),
        ('educational_datasets', 'asap2'),
        ('educational_datasets', 'ellipse'),
        ('educational_datasets', 'mohler'),
        ('educational_datasets', 'ptasag'),
        ('educational_datasets', 'arasag'),
        
        # Data flow - direct path
        ('educational_datasets', 'model_response'),
        ('model_response', 'grade_parser'),
        ('grade_parser', 'metrics'),
    ]
    
    # Draw connections with better routing
    for start, end in connections:
        start_pos = components[start]['pos']
        end_pos = components[end]['pos']
        
        # Calculate connection points (edges of boxes)
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Get box sizes
        start_w, start_h = components[start]['size']
        end_w, end_h = components[end]['size']
        
        # Calculate connection points on box edges
        if start_y > end_y:  # Vertical connection downward
            start_conn = (start_x, start_y - start_h/2)
            end_conn = (end_x, end_y + end_h/2)
        elif start_y < end_y:  # Vertical connection upward
            start_conn = (start_x, start_y + start_h/2)
            end_conn = (end_x, end_y - end_h/2)
        elif start_x < end_x:  # Horizontal connection rightward
            start_conn = (start_x + start_w/2, start_y)
            end_conn = (end_x - end_w/2, end_y)
        else:  # Horizontal connection leftward
            start_conn = (start_x - start_w/2, start_y)
            end_conn = (end_x + end_w/2, end_y)
        
        # Create connection line
        connection = ConnectionPatch(
            start_conn, end_conn,
            "data", "data",
            arrowstyle="->",
            shrinkA=0, shrinkB=0,
            mutation_scale=15,
            fc="gray", ec="gray",
            linewidth=1.2,
            alpha=0.7
        )
        ax.add_patch(connection)
    
    # Draw components AFTER connections (on top)
    for name, props in components.items():
        x, y = props['pos']
        w, h = props['size']
        color = colors[props['color']]
        border_color = colors[f'border_{props["color"]}']
        
        # Create rounded rectangle
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor=border_color,
            linewidth=2
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y, props['text'], 
                ha='center', va='center', 
                fontsize=8, fontweight='bold',
                wrap=True)
    
    # Add title
    ax.text(8, 17.5, 'MentorEval Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Add legend (positioned to not cover the title)
    legend_elements = [
        patches.Patch(color=legend_colors['lighteval'], label='LightEval Core Components'),
        patches.Patch(color=legend_colors['custom'], label='MentorEval Extensions'),
        patches.Patch(color=legend_colors['dataset'], label='Educational Datasets'),
        patches.Patch(color=legend_colors['metric'], label='Evaluation Metrics')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.95),
              fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Adjust subplot to reduce bottom margin
    plt.subplots_adjust(bottom=0.05)
    
    return fig

def main():
    """Generate and save the architecture diagram to webpage/assets/."""
    
    print("🏗️  Generating MentorEval Architecture Diagram")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("../webpage/assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate diagram
    print("🔄 Creating architecture diagram...")
    fig = create_architecture_diagram()
    
    # Save as PNG with custom bbox to trim bottom margin
    output_path = output_dir / "mentoreval_architecture.png"
    fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ Saved: {output_path}")
    
    print(f"\n🎉 Architecture diagram generated successfully!")
    print(f"📁 Location: {output_path.absolute()}")

if __name__ == "__main__":
    main()
