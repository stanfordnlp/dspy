#!/usr/bin/env python3
"""
Manual trajectory analysis for GEPA results when detailed_results aren't available.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analysis import ExperimentAnalyzer


def create_manual_gepa_analysis():
    """Create manual GEPA analysis based on observed results."""
    
    # Based on the log output, we observed:
    # Baseline: 55.74% (0.5574)
    # Candidate 1 (head of state instructions): 55.87% (0.5587)  
    # Candidate 2 (Clydesdale instructions): 57.82% (0.5782) - BEST
    
    candidates_data = [
        {"iteration": 0, "score": 0.5574, "name": "Baseline (Original)", "is_best": False},
        {"iteration": 1, "score": 0.5587, "name": "Head of State Instructions", "is_best": False},
        {"iteration": 2, "score": 0.5782, "name": "Clydesdale Plural Form Instructions", "is_best": True},
    ]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Optimization trajectory
    iterations = [c["iteration"] for c in candidates_data]
    scores = [c["score"] for c in candidates_data]
    colors = ['blue' if not c["is_best"] else 'red' for c in candidates_data]
    
    # Plot trajectory line
    ax1.plot(iterations, scores, 'o-', linewidth=2, markersize=8, color='darkblue', alpha=0.7)
    
    # Highlight best candidate
    best_candidate = next(c for c in candidates_data if c["is_best"])
    ax1.scatter([best_candidate["iteration"]], [best_candidate["score"]], 
               color='red', s=150, marker='*', edgecolor='black', linewidth=2,
               label='Best Candidate', zorder=5)
    
    # Annotate points
    for i, c in enumerate(candidates_data):
        ax1.annotate(f"C{i}: {c['score']:.4f}", 
                    (c["iteration"], c["score"]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, ha='left')
    
    ax1.set_xlabel("Candidate Iteration")
    ax1.set_ylabel("Validation Score")
    ax1.set_title("GEPA Optimization Trajectory\n(HotPotQA with 100 train, 20 dev)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0.55, 0.59)
    
    # Right plot: Performance comparison
    baseline_score = candidates_data[0]["score"]
    best_score = best_candidate["score"]
    improvement = best_score - baseline_score
    
    categories = ['Baseline', 'GEPA Optimized']
    values = [baseline_score * 100, best_score * 100]
    colors_bar = ['lightblue', 'lightgreen']
    
    bars = ax2.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1)
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("Performance Improvement")
    ax2.set_ylim(55, 59)
    
    # Add improvement annotation
    ax2.annotate(f'+{improvement*100:.2f}%\n(+{improvement/baseline_score*100:.1f}% relative)', 
                xy=(1, best_score * 100), xytext=(0.5, best_score * 100 + 1),
                ha='center', fontsize=12, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = "results/reports/gepa_manual_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Manual GEPA analysis saved to: {output_path}")
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("GEPA OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Dataset: HotPotQA (100 train, 20 validation)")
    print(f"Optimizer: GEPA (max_metric_calls=50)")
    print(f"Program: mlflow_base_prompt")
    print()
    print("CANDIDATE EXPLORATION:")
    for i, c in enumerate(candidates_data):
        marker = " ★ BEST" if c["is_best"] else ""
        print(f"  Candidate {i}: {c['score']:.4f} ({c['score']*100:.2f}%) - {c['name']}{marker}")
    print()
    print(f"IMPROVEMENT: {improvement:.4f} (+{improvement/baseline_score*100:.1f}% relative)")
    print(f"CONVERGENCE: Found improvement in {len(candidates_data)-1} iterations")
    
    print("\nINSTRUCTION EVOLUTION:")
    print("1. Baseline: Generic QA instructions")
    print("2. Candidate 1: Added head of state position extraction rules")
    print("3. Candidate 2: Added Clydesdale plural form handling (WINNING STRATEGY)")
    
    print(f"\nGEPA successfully identified specific linguistic patterns and")
    print(f"developed targeted instruction improvements!")


if __name__ == "__main__":
    create_manual_gepa_analysis()