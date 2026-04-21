#!/usr/bin/env python3
"""
Script to run multiple experiments and generate comparative analysis.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from core.experiment import ExperimentRunner
from core.analysis import ExperimentAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments and generate comparative analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "configs",
        nargs="+",
        help="Paths to experiment configuration files"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="results/reports",
        help="Directory to save analysis reports"
    )
    
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip running experiments, just analyze existing results"
    )
    
    args = parser.parse_args()
    
    experiment_names = []
    
    try:
        if not args.skip_experiments:
            # Run all experiments
            print("=" * 60)
            print("RUNNING EXPERIMENTS")
            print("=" * 60)
            
            for config_path in args.configs:
                print(f"\n>>> Running experiment: {config_path}")
                
                # Load and run experiment
                config = load_config(config_path)
                experiment_names.append(config.name)
                
                runner = ExperimentRunner(config)
                results = runner.run()
                
                print(f"✓ Experiment {config.name} completed")
                print(f"  Results: {results['results_path']}")
        else:
            # Extract experiment names from config files
            for config_path in args.configs:
                config = load_config(config_path)
                experiment_names.append(config.name)
        
        # Run comparative analysis
        print("\n" + "=" * 60)
        print("GENERATING COMPARATIVE ANALYSIS")
        print("=" * 60)
        
        analyzer = ExperimentAnalyzer()
        
        # Load all experiment results
        experiment_results = []
        for exp_name in experiment_names:
            try:
                results = analyzer.load_experiment_results(exp_name)
                experiment_results.append((exp_name, results))
                print(f"✓ Loaded results for: {exp_name}")
            except FileNotFoundError:
                print(f"✗ Could not find results for: {exp_name}")
        
        if not experiment_results:
            print("No experiment results found for analysis.")
            sys.exit(1)
        
        # Generate trajectory comparison plot
        print("\n>>> Generating optimization trajectory plot...")
        plot_path = Path(args.output_dir) / "comparative_trajectory.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        
        analyzer.plot_optimization_trajectory(
            experiment_results, 
            str(plot_path),
            figsize=(14, 10)
        )
        
        # Generate individual GEPA analyses
        for exp_name, results in experiment_results:
            if results["config"]["optimizer"]["name"] == "gepa":
                print(f">>> Generating GEPA analysis for {exp_name}...")
                try:
                    gepa_plot_path = Path(args.output_dir) / f"gepa_detailed_{exp_name}.png"
                    analyzer.plot_gepa_candidate_exploration(exp_name, str(gepa_plot_path))
                except Exception as e:
                    print(f"Warning: GEPA analysis failed for {exp_name}: {e}")
        
        # Generate comprehensive HTML report
        print(">>> Generating HTML report...")
        report_path = analyzer.generate_experiment_report(
            experiment_results,
            args.output_dir
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        print(f"\nGenerated Files:")
        print(f"  📊 Trajectory Plot: {plot_path}")
        print(f"  📋 HTML Report: {report_path}")
        
        # Print quick comparison
        print(f"\nQuick Results Summary:")
        for exp_name, results in experiment_results:
            baseline = results.get("baseline_results", {})
            optimized = results.get("optimization_results", {})
            
            baseline_score = baseline.get("avg_score", 0)
            optimized_score = optimized.get("avg_score", baseline_score)
            improvement = optimized_score - baseline_score
            
            optimizer = results["config"]["optimizer"]["name"]
            print(f"  {exp_name:20} ({optimizer:8}): {baseline_score:.3f} → {optimized_score:.3f} ({improvement:+.3f})")
        
        print(f"\n✓ Open {report_path} in your browser to view the full analysis!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()