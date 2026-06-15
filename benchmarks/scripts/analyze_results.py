#!/usr/bin/env python3
"""
Script to analyze and visualize experiment results.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analysis import ExperimentAnalyzer, analyze_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize prompt optimization experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Names of experiments to analyze"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/logs",
        help="Directory containing experiment results"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="results/reports",
        help="Directory to save analysis reports"
    )
    
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (useful for headless environments)"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = ExperimentAnalyzer(args.results_dir)
        
        # Load experiment results
        experiment_results = []
        for exp_name in args.experiments:
            try:
                results = analyzer.load_experiment_results(exp_name)
                experiment_results.append((exp_name, results))
                print(f"✓ Loaded results for: {exp_name}")
            except FileNotFoundError:
                print(f"✗ Could not find results for: {exp_name}")
        
        if not experiment_results:
            print("No experiment results found to analyze.")
            sys.exit(1)
        
        print(f"\nAnalyzing {len(experiment_results)} experiments...")
        
        # Generate trajectory comparison plot
        trajectory_plot_path = Path(args.output_dir) / "trajectory_comparison.png"
        trajectory_plot_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig = analyzer.plot_optimization_trajectory(
            experiment_results, 
            str(trajectory_plot_path)
        )
        
        if not args.no_show:
            import matplotlib.pyplot as plt
            plt.show()
        
        # Generate GEPA-specific analysis for GEPA experiments
        for exp_name, results in experiment_results:
            if results["config"]["optimizer"]["name"] == "gepa":
                try:
                    gepa_plot_path = Path(args.output_dir) / f"gepa_analysis_{exp_name}.png"
                    gepa_fig = analyzer.plot_gepa_candidate_exploration(
                        exp_name, 
                        str(gepa_plot_path)
                    )
                    
                    if not args.no_show:
                        import matplotlib.pyplot as plt
                        plt.show()
                        
                except Exception as e:
                    print(f"Warning: Could not generate GEPA analysis for {exp_name}: {e}")
        
        # Generate comprehensive HTML report
        report_path = analyzer.generate_experiment_report(
            experiment_results, 
            args.output_dir
        )
        
        print(f"\n✓ Analysis complete!")
        print(f"✓ Trajectory plot: {trajectory_plot_path}")
        print(f"✓ HTML report: {report_path}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()