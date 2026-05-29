"""
Visualization utilities for model comparison results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json

class ModelComparisonVisualizer:
    """Class to create visualizations for model comparison results."""

    def __init__(self, results_path: str = "results"):
        """
        Initialize visualizer with results path.

        Args:
            results_path: Path to results directory
        """
        self.results_path = Path(results_path)
        self.results = None
        self.summary_df = None

    def load_results(self) -> None:
        """Load results from saved files."""
        # Load complete results
        results_file = self.results_path / "complete_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.results = json.load(f)

        # Load summary DataFrame
        summary_file = self.results_path / "model_comparison_summary.csv"
        if summary_file.exists():
            self.summary_df = pd.read_csv(summary_file)

    def plot_metric_comparison(self, metrics: List[str] = None, save_path: Optional[str] = None):
        """
        Create bar plots comparing models across specified metrics.

        Args:
            metrics: List of metrics to plot (defaults to main metrics)
            save_path: Path to save the plot
        """
        if self.summary_df is None:
            self.load_results()

        if metrics is None:
            metrics = ['mae', 'rmse', 'r2', 'accuracy', 'f1_score']

        # Filter available metrics
        available_metrics = [m for m in metrics if m in self.summary_df.columns]

        if not available_metrics:
            print("No valid metrics found in data")
            return

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            ax = axes[i]

            # Create bar plot
            bars = ax.bar(self.summary_df['Model'], self.summary_df[metric])

            # Customize plot
            ax.set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12)
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)

            # Color bars by performance (for metrics where higher is better)
            if metric in ['r2', 'accuracy', 'precision', 'recall', 'f1_score']:
                colors = ['green' if x == max(self.summary_df[metric]) else 'lightblue'
                         for x in self.summary_df[metric]]
            else:  # For metrics where lower is better
                colors = ['green' if x == min(self.summary_df[metric]) else 'lightcoral'
                         for x in self.summary_df[metric]]

            for bar, color in zip(bars, colors):
                bar.set_color(color)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metric comparison plot saved to {save_path}")

        plt.show()

    def plot_training_time_comparison(self, save_path: Optional[str] = None):
        """
        Create a comparison of training times.

        Args:
            save_path: Path to save the plot
        """
        if self.summary_df is None:
            self.load_results()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Training time comparison
        bars1 = ax1.bar(self.summary_df['Model'], self.summary_df['Training_Time_s'])
        ax1.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Training Time (seconds)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontsize=10)

        # Inference time comparison
        bars2 = ax2.bar(self.summary_df['Model'], self.summary_df['Inference_Time_s'])
        ax2.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Inference Time (seconds)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}s', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Time comparison plot saved to {save_path}")

        plt.show()

    def plot_performance_radar(self, save_path: Optional[str] = None):
        """
        Create radar chart comparing model performance across multiple metrics.

        Args:
            save_path: Path to save the plot
        """
        if self.summary_df is None:
            self.load_results()

        try:
            # Select metrics for radar chart (normalize to 0-1 scale)
            metrics = ['mae', 'rmse', 'r2', 'accuracy', 'f1_score']
            available_metrics = [m for m in metrics if m in self.summary_df.columns]

            if len(available_metrics) < 3:
                print("Not enough metrics available for radar chart")
                return

            # Normalize metrics (0-1 scale, where 1 is best)
            normalized_data = self.summary_df[['Model'] + available_metrics].copy()

            for metric in available_metrics:
                if metric in ['r2', 'accuracy', 'precision', 'recall', 'f1_score']:
                    # Higher is better - normalize to 0-1
                    max_val = normalized_data[metric].max()
                    min_val = normalized_data[metric].min()
                    if max_val != min_val:
                        normalized_data[metric] = (normalized_data[metric] - min_val) / (max_val - min_val)
                    else:
                        normalized_data[metric] = 1.0
                else:
                    # Lower is better - inverse normalize
                    max_val = normalized_data[metric].max()
                    min_val = normalized_data[metric].min()
                    if max_val != min_val:
                        normalized_data[metric] = 1 - (normalized_data[metric] - min_val) / (max_val - min_val)
                    else:
                        normalized_data[metric] = 1.0

            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

            # Number of variables
            N = len(available_metrics)

            # Compute angle for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle

            # Colors for each model
            colors = ['red', 'blue', 'green', 'orange', 'purple']

            for i, (_, row) in enumerate(normalized_data.iterrows()):
                values = row[available_metrics].values.tolist()
                values += values[:1]  # Complete the circle

                ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'],
                       color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

            # Add metric labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.upper() for m in available_metrics])

            # Set y-axis limits
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])

            ax.set_title('Model Performance Comparison (Normalized)', size=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            ax.grid(True)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Radar chart saved to {save_path}")

            plt.show()

        except Exception as e:
            print(f"Could not create radar chart: {e}")

    def plot_cross_validation_results(self, save_path: Optional[str] = None):
        """
        Create box plots for cross-validation results.

        Args:
            save_path: Path to save the plot
        """
        if self.results is None:
            self.load_results()

        cv_results = self.results.get('cross_validation_results')
        if not cv_results:
            print("No cross-validation results available")
            return

        # Prepare data for plotting
        plot_data = []
        for model_name, model_cv in cv_results.items():
            for metric_name, scores in model_cv.items():
                for score in scores:
                    plot_data.append({
                        'Model': model_name,
                        'Metric': metric_name,
                        'Score': score
                    })

        if not plot_data:
            print("No cross-validation data to plot")
            return

        cv_df = pd.DataFrame(plot_data)

        # Select main metrics for plotting
        main_metrics = ['mae', 'rmse', 'r2', 'accuracy', 'f1_score']
        available_metrics = [m for m in main_metrics if m in cv_df['Metric'].values]

        if not available_metrics:
            print("No main metrics available for cross-validation plotting")
            return

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            metric_data = cv_df[cv_df['Metric'] == metric]

            # Create box plot
            models = metric_data['Model'].unique()
            data_by_model = [metric_data[metric_data['Model'] == model]['Score'].values
                           for model in models]

            box_plot = ax.boxplot(data_by_model, labels=models, patch_artist=True)

            # Customize box plot
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_title(f'{metric.upper()} - Cross Validation', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cross-validation plot saved to {save_path}")

        plt.show()

    def create_summary_report(self, output_path: Optional[str] = None):
        """
        Create a comprehensive summary report with multiple visualizations.

        Args:
            output_path: Path to save the report plots
        """
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

        print("Creating comprehensive visualization report...")

        # Main metrics comparison
        self.plot_metric_comparison(
            save_path=str(output_path / "metric_comparison.png") if output_path else None
        )

        # Training time comparison
        self.plot_training_time_comparison(
            save_path=str(output_path / "time_comparison.png") if output_path else None
        )

        # Performance radar chart
        self.plot_performance_radar(
            save_path=str(output_path / "performance_radar.png") if output_path else None
        )

        # Cross-validation results
        self.plot_cross_validation_results(
            save_path=str(output_path / "cross_validation.png") if output_path else None
        )

        print("Visualization report completed!")

def main():
    """Main function to generate visualizations."""
    visualizer = ModelComparisonVisualizer()

    try:
        visualizer.load_results()
        visualizer.create_summary_report(output_path="results/plots")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Make sure you have run the model comparison script first!")

if __name__ == "__main__":
    main()