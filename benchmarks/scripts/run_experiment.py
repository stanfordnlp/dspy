#!/usr/bin/env python3
"""
Main script to run prompt optimization experiments.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from core.experiment import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run prompt optimization experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment configuration file"
    )
    
    parser.add_argument(
        "--no-analysis",
        action="store_true", 
        help="Skip automatic analysis after experiment"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        print(f"Loaded experiment: {config.name}")
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {results['results_path']}")
        
        # Run automatic analysis
        if not args.no_analysis:
            print("\n" + "="*60)
            print("RUNNING POST-EXPERIMENT ANALYSIS")
            print("="*60)
            
            try:
                from core.analysis import analyze_experiments
                analyze_experiments(config.name)
            except Exception as e:
                print(f"Warning: Could not run analysis: {e}")
                print(f"You can run analysis later with:")
                print(f"python scripts/analyze_results.py {config.name}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()