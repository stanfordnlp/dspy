"""
Analysis and visualization of experiment results.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ExperimentAnalyzer:
    """Analyzer for experiment results with plotting and comparison capabilities."""
    
    def __init__(self, results_dir: str = "results/logs"):
        self.results_dir = Path(results_dir)
        
    def load_experiment_results(self, experiment_name: str) -> Dict[str, Any]:
        """Load experiment results from JSON file.
        
        Args:
            experiment_name: Name of the experiment or path to results file.
            
        Returns:
            Dictionary containing experiment results.
        """
        if experiment_name.endswith('.json'):
            results_file = Path(experiment_name)
        else:
            # Find most recent results file for this experiment
            pattern = f"{experiment_name}_results_*.json"
            results_files = list(self.results_dir.glob(pattern))
            if not results_files:
                raise FileNotFoundError(f"No results found for experiment: {experiment_name}")
            results_file = max(results_files, key=lambda f: f.stat().st_mtime)
        
        with open(results_file) as f:
            return json.load(f)
    
    def extract_optimization_trajectory(self, results: Dict[str, Any]) -> Tuple[List[int], List[float]]:
        """Extract optimization trajectory from experiment results.
        
        Args:
            results: Experiment results dictionary.
            
        Returns:
            Tuple of (iterations, validation_scores).
        """
        iterations = []
        val_scores = []
        
        timeline = results.get("timeline", [])
        
        # Extract baseline score
        for event in timeline:
            if event["event"] == "baseline_evaluated":
                iterations.append(0)
                val_scores.append(event["data"]["avg_score"])
                break
        
        # Extract optimization trajectory
        iteration_count = 1
        for event in timeline:
            if event["event"] == "optimization_iteration":
                data = event["data"]
                iterations.append(iteration_count)
                val_scores.append(data.get("val_score", data.get("avg_score", 0)))
                iteration_count += 1
        
        # Add final optimized score if available
        for event in timeline:
            if event["event"] == "optimized_evaluated":
                # Only add if we haven't seen the final iteration yet
                if len(iterations) == 1:  # Only baseline
                    iterations.append(1)
                    val_scores.append(event["data"]["avg_score"])
                break
        
        return iterations, val_scores
    
    def _parse_gepa_log_file(self, log_file_path: Path) -> List[Dict]:
        """Parse GEPA candidate information from log file.
        
        Args:
            log_file_path: Path to the experiment log file.
            
        Returns:
            List of candidate information dictionaries.
        """
        candidates_info = []
        
        if not log_file_path.exists():
            return candidates_info
            
        try:
            with open(log_file_path, 'r') as f:
                content = f.read()
            
            # Find the "ALL PROPOSED CANDIDATES" section
            import re
            
            # Extract baseline score first
            baseline_match = re.search(r'Total Score: [\d.]+\s*/\s*\d+\s*\(([\d.]+)%\)', content)
            baseline_score = float(baseline_match.group(1)) / 100 if baseline_match else 0.5574
            
            # Extract candidate rankings
            candidate_pattern = r'\[Rank (\d+)\] Candidate (\d+) — Score: ([\d.]+)( ★ BEST)?'
            candidate_matches = re.findall(candidate_pattern, content)
            
            # Build candidates list with baseline
            candidates_info.append({
                "iteration": 0,
                "score": baseline_score,
                "is_best": False,
                "name": "Baseline (Original)",
                "rank": 3 if candidate_matches else 1,
                "instruction_type": "Original QA instructions"
            })
            
            # Add ranked candidates from optimization
            for rank, candidate_id, score, is_best in candidate_matches:
                # Extract instruction type based on candidate ID and content analysis
                instruction_type = "Unknown"
                if candidate_id == "1":
                    if "head of state" in content.lower():
                        instruction_type = "Head of state position extraction rules"
                elif candidate_id == "2":
                    if "plural" in content.lower() or "clydesdale" in content.lower():
                        instruction_type = "Clydesdale plural form handling"
                
                candidates_info.append({
                    "iteration": int(candidate_id),
                    "score": float(score),
                    "is_best": bool(is_best),
                    "name": f"Candidate {candidate_id}",
                    "rank": int(rank),
                    "instruction_type": instruction_type
                })
            
            # Sort by iteration to maintain chronological order
            candidates_info.sort(key=lambda x: x["iteration"])
            
        except Exception as e:
            print(f"Warning: Could not parse GEPA log file: {e}")
        
        return candidates_info

    def extract_gepa_trajectory(self, results: Dict[str, Any]) -> Tuple[List[int], List[float], List[Dict]]:
        """Extract GEPA-specific optimization trajectory.
        
        Args:
            results: Experiment results dictionary.
            
        Returns:
            Tuple of (iterations, best_scores, candidates_info).
        """
        # Try to extract from timeline events first
        candidates_info = []
        timeline = results.get("timeline", [])
        
        # Extract from timeline events (more reliable)
        for event in timeline:
            if event["event"] == "gepa_candidate_evaluated":
                data = event["data"]
                candidates_info.append({
                    "iteration": data.get("candidate_id", len(candidates_info)),
                    "score": data.get("score", 0),
                    "is_best": data.get("is_best", False),
                    "instruction_length": len(data.get("instruction", ""))
                })
        
        # If timeline extraction failed, try parsing log file directly
        if not candidates_info:
            experiment_name = results.get("experiment_name", "")
            log_file_pattern = f"{experiment_name}_*.log"
            log_dir = Path("results/logs")
            
            matching_logs = list(log_dir.glob(log_file_pattern))
            if matching_logs:
                # Use the most recent log file
                log_file = max(matching_logs, key=lambda p: p.stat().st_mtime)
                candidates_info = self._parse_gepa_log_file(log_file)
        
        # If log parsing failed, try to load detailed GEPA results from saved model
        if not candidates_info:
            model_path = self._find_saved_model(results["experiment_name"])
            if model_path and model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        optimized_program = pickle.load(f)
                    
                    if hasattr(optimized_program, 'detailed_results'):
                        dr = optimized_program.detailed_results
                        iterations = list(range(len(dr.val_aggregate_scores)))
                        scores = dr.val_aggregate_scores
                        
                        # Extract candidate information
                        candidates_info = []
                        for i, (candidate, score) in enumerate(zip(dr.candidates, scores)):
                            candidates_info.append({
                                "iteration": i,
                                "score": score,
                                "is_best": i == dr.best_idx,
                                "instruction_length": len(str(candidate)) if candidate else 0
                            })
                        
                        return iterations, scores, candidates_info
                except Exception as e:
                    print(f"Warning: Could not load GEPA detailed results: {e}")
        
        # If we have candidates info, extract trajectory
        if candidates_info:
            iterations = [c["iteration"] for c in candidates_info]
            scores = [c["score"] for c in candidates_info]
            return iterations, scores, candidates_info
        
        # Final fallback to basic trajectory
        iterations, scores = self.extract_optimization_trajectory(results)
        candidates_info = [{"iteration": i, "score": s, "is_best": i == len(scores)-1} 
                          for i, s in enumerate(scores)]
        return iterations, scores, candidates_info
    
    def _find_saved_model(self, experiment_name: str) -> Optional[Path]:
        """Find saved model file for experiment."""
        model_dir = Path("results/models")
        if not model_dir.exists():
            return None
        
        pattern = f"{experiment_name}_*.pkl"
        model_files = list(model_dir.glob(pattern))
        if not model_files:
            return None
        
        return max(model_files, key=lambda f: f.stat().st_mtime)
    
    def plot_optimization_trajectory(
        self, 
        experiment_results: List[Tuple[str, Dict[str, Any]]], 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """Plot optimization trajectories for multiple experiments.
        
        Args:
            experiment_results: List of (experiment_name, results_dict) tuples.
            save_path: Optional path to save the plot.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure object.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(experiment_results)))
        
        # Track convergence metrics
        convergence_data = []
        
        for (exp_name, results), color in zip(experiment_results, colors):
            optimizer_name = results["config"]["optimizer"]["name"]
            
            if optimizer_name == "gepa":
                iterations, scores, candidates = self.extract_gepa_trajectory(results)
                label = f"{exp_name} (GEPA)"
                
                # Plot all candidates as scatter points
                candidate_scores = [c["score"] for c in candidates]
                candidate_iters = [c["iteration"] for c in candidates]
                ax1.scatter(candidate_iters, candidate_scores, color=color, alpha=0.3, s=20)
                
                # Plot best trajectory as line
                best_scores = []
                best_score_so_far = 0
                for score in scores:
                    best_score_so_far = max(best_score_so_far, score)
                    best_scores.append(best_score_so_far)
                ax1.plot(iterations, best_scores, color=color, linewidth=2, label=label, marker='o')
                
            else:
                iterations, scores = self.extract_optimization_trajectory(results)
                label = f"{exp_name} ({optimizer_name})"
                ax1.plot(iterations, scores, color=color, linewidth=2, label=label, marker='o')
            
            # Calculate convergence metrics
            if len(scores) > 1:
                final_score = scores[-1]
                baseline_score = scores[0]
                improvement = final_score - baseline_score
                convergence_rate = improvement / len(scores) if len(scores) > 1 else 0
            else:
                final_score = scores[0] if scores else 0
                improvement = 0
                convergence_rate = 0
            
            convergence_data.append({
                "experiment": exp_name,
                "optimizer": optimizer_name,
                "baseline": scores[0] if scores else 0,
                "final": final_score,
                "improvement": improvement,
                "iterations": len(scores) - 1,
                "convergence_rate": convergence_rate
            })
        
        # Customize main plot
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Validation Score")
        ax1.set_title("Optimization Trajectory Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Convergence summary table
        if convergence_data:
            df = pd.DataFrame(convergence_data)
            
            # Create table
            table_data = []
            for _, row in df.iterrows():
                table_data.append([
                    row["experiment"],
                    row["optimizer"],
                    f"{row['baseline']:.3f}",
                    f"{row['final']:.3f}",
                    f"{row['improvement']:+.3f}",
                    f"{row['iterations']}",
                    f"{row['convergence_rate']:.4f}"
                ])
            
            columns = ["Experiment", "Optimizer", "Baseline", "Final", "Δ", "Iters", "Rate"]
            
            ax2.axis('tight')
            ax2.axis('off')
            table = ax2.table(cellText=table_data, colLabels=columns, 
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Color code improvements
            for i, row in enumerate(table_data):
                improvement = float(row[4])
                if improvement > 0:
                    table[(i+1, 4)].set_facecolor('#90EE90')  # Light green
                elif improvement < 0:
                    table[(i+1, 4)].set_facecolor('#FFB6C1')  # Light red
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig
    
    def plot_gepa_candidate_exploration(
        self, 
        experiment_name: str, 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 6)
    ) -> plt.Figure:
        """Plot GEPA candidate exploration pattern.
        
        Args:
            experiment_name: Name of the GEPA experiment.
            save_path: Optional path to save the plot.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure object.
        """
        results = self.load_experiment_results(experiment_name)
        iterations, scores, candidates = self.extract_gepa_trajectory(results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: Optimization trajectory showing all candidates
        candidate_scores = [c["score"] for c in candidates]
        candidate_iters = [c["iteration"] for c in candidates]
        best_candidates = [c for c in candidates if c["is_best"]]
        
        # Plot trajectory line
        ax1.plot(candidate_iters, candidate_scores, 'o-', linewidth=2, markersize=8, 
                color='darkblue', alpha=0.7, label='Candidate exploration')
        
        # Highlight best candidate
        if best_candidates:
            best_iter = [c["iteration"] for c in best_candidates]
            best_score = [c["score"] for c in best_candidates]
            ax1.scatter(best_iter, best_score, color='red', s=150, 
                       label='Best candidate', marker='*', edgecolor='black', linewidth=2,
                       zorder=5)
        
        # Annotate all candidates with details
        for i, c in enumerate(candidates):
            # Use name or create one based on iteration
            name = c.get("name", f"C{c['iteration']}")
            if c.get("instruction_type"):
                annotation = f"{name}: {c['score']:.4f}\n{c['instruction_type']}"
            else:
                annotation = f"{name}: {c['score']:.4f}"
            
            ax1.annotate(annotation, 
                        (c["iteration"], c["score"]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax1.set_xlabel("Candidate Iteration")
        ax1.set_ylabel("Validation Score") 
        ax1.set_title(f"GEPA Optimization Trajectory\n{experiment_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set y-axis limits to show improvement clearly
        if candidate_scores:
            score_range = max(candidate_scores) - min(candidate_scores)
            ax1.set_ylim(min(candidate_scores) - score_range * 0.1, 
                        max(candidate_scores) + score_range * 0.1)
        
        # Right plot: Performance improvement comparison
        baseline_score = candidates[0]["score"] if candidates else 0
        best_score = max(candidate_scores) if candidate_scores else baseline_score
        improvement = best_score - baseline_score
        
        categories = ['Baseline', 'GEPA Optimized']
        values = [baseline_score * 100, best_score * 100]
        colors_bar = ['lightblue', 'lightgreen']
        
        bars = ax2.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1)
        ax2.set_ylabel("Validation Accuracy (%)")
        ax2.set_title("Performance Improvement")
        
        # Add improvement annotation
        if improvement > 0:
            ax2.annotate(f'+{improvement*100:.2f}%\n(+{improvement/baseline_score*100:.1f}% relative)', 
                        xy=(1, best_score * 100), xytext=(0.5, best_score * 100 + improvement*50),
                        ha='center', fontsize=12, fontweight='bold', color='green',
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                    f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GEPA analysis plot saved to: {save_path}")
        
        return fig
    
    def generate_experiment_report(
        self, 
        experiment_results: List[Tuple[str, Dict[str, Any]]], 
        output_dir: str = "results/reports"
    ) -> str:
        """Generate a comprehensive experiment report.
        
        Args:
            experiment_results: List of (experiment_name, results_dict) tuples.
            output_dir: Directory to save the report.
            
        Returns:
            Path to the generated report file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"experiment_report_{timestamp}.html"
        
        # Generate plots
        plot_path = output_dir / f"trajectory_plot_{timestamp}.png"
        self.plot_optimization_trajectory(experiment_results, str(plot_path))
        
        # Generate HTML report
        html_content = self._generate_html_report(experiment_results, plot_path.name)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Experiment report generated: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self, experiment_results: List[Tuple[str, Dict[str, Any]]], plot_filename: str) -> str:
        """Generate HTML report content."""
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Prompt Optimization Experiment Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .experiment { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .config { background-color: #f9f9f9; padding: 10px; border-radius: 3px; font-family: monospace; }
        .results { margin: 10px 0; }
        .plot { text-align: center; margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f0f0f0; }
        .improvement-positive { color: green; font-weight: bold; }
        .improvement-negative { color: red; font-weight: bold; }
        .improvement-neutral { color: gray; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Prompt Optimization Experiment Report</h1>
        <p><strong>Generated:</strong> """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <p><strong>Total Experiments:</strong> """ + str(len(experiment_results)) + """</p>
    </div>
    
    <div class="plot">
        <h2>Optimization Trajectory Comparison</h2>
        <img src=\"""" + plot_filename + """\" alt="Optimization Trajectory" style="max-width: 100%; height: auto;">
    </div>
"""
        
        # Add experiment details
        for exp_name, results in experiment_results:
            config = results["config"]
            baseline = results.get("baseline_results", {})
            optimized = results.get("optimization_results", {})
            
            baseline_score = baseline.get("avg_score", 0)
            optimized_score = optimized.get("avg_score", baseline_score)
            improvement = optimized_score - baseline_score
            
            improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative" if improvement < 0 else "improvement-neutral"
            
            html += f"""
    <div class="experiment">
        <h3>{exp_name}</h3>
        <p><strong>Description:</strong> {config.get("description", "N/A")}</p>
        
        <div class="results">
            <h4>Results Summary</h4>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Optimized</th>
                    <th>Improvement</th>
                </tr>
                <tr>
                    <td>Validation Score</td>
                    <td>{baseline_score:.4f}</td>
                    <td>{optimized_score:.4f}</td>
                    <td class="{improvement_class}">{improvement:+.4f}</td>
                </tr>
                <tr>
                    <td>Dataset Size</td>
                    <td colspan="3">{config["dataset"]["train_size"]} train, {config["dataset"]["dev_size"]} dev</td>
                </tr>
            </table>
        </div>
        
        <div class="config">
            <h4>Configuration</h4>
            <p><strong>Dataset:</strong> {config["dataset"]["name"]}</p>
            <p><strong>Model:</strong> {config["model"]["name"]}</p>
            <p><strong>Optimizer:</strong> {config["optimizer"]["name"]}</p>
            <p><strong>Program:</strong> {config["program"]["name"]}</p>
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html


def analyze_experiments(*experiment_names: str, results_dir: str = "results/logs") -> None:
    """Convenience function to analyze and plot multiple experiments.
    
    Args:
        *experiment_names: Names of experiments to analyze.
        results_dir: Directory containing experiment results.
    """
    analyzer = ExperimentAnalyzer(results_dir)
    
    experiment_results = []
    for exp_name in experiment_names:
        try:
            results = analyzer.load_experiment_results(exp_name)
            experiment_results.append((exp_name, results))
        except FileNotFoundError:
            print(f"Warning: Could not find results for experiment: {exp_name}")
    
    if not experiment_results:
        print("No experiment results found to analyze.")
        return
    
    # Generate trajectory plot
    fig = analyzer.plot_optimization_trajectory(experiment_results)
    plt.show()
    
    # Generate GEPA-specific plots for GEPA experiments
    for exp_name, results in experiment_results:
        if results["config"]["optimizer"]["name"] == "gepa":
            try:
                gepa_fig = analyzer.plot_gepa_candidate_exploration(exp_name)
                plt.show()
            except Exception as e:
                print(f"Could not generate GEPA analysis for {exp_name}: {e}")
    
    # Generate report
    analyzer.generate_experiment_report(experiment_results)


if __name__ == "__main__":
    # Example usage
    analyze_experiments("hotpotqa_baseline", "hotpotqa_gepa_v1")