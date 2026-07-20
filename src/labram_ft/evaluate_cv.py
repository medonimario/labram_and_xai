import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_cv_logs(cv_dir):
    """
    Parses log.txt files from all fold directories within cv_dir.
    Returns:
        epochs_data: dict of aligned metrics {metric_name: {epoch: [val_fold0, val_fold1, ...]}}
        best_test_metrics: list of dicts containing the best test metrics per fold
    """
    epochs_data = defaultdict(lambda: defaultdict(list))
    best_test_metrics = []

    fold_dirs = [d for d in os.listdir(cv_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(cv_dir, d))]
    
    if not fold_dirs:
        print(f"No 'fold_X' directories found in {cv_dir}")
        return None, None

    for fold_name in sorted(fold_dirs):
        log_path = os.path.join(cv_dir, fold_name, 'log.txt')
        if not os.path.exists(log_path):
            print(f"Skipping {fold_name}: No log.txt found.")
            continue

        fold_best_val_acc = -1
        fold_best_metrics = {}

        try:
            with open(log_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        epoch = data['epoch']
                        
                        # Store data for curve plotting
                        for key, value in data.items():
                            if isinstance(value, (int, float)):
                                epochs_data[key][epoch].append(value)
                        
                        # Track the best epoch for this fold based on Validation Balanced Accuracy
                        val_acc = data.get('val_balanced_accuracy', -1)
                        if val_acc > fold_best_val_acc:
                            fold_best_val_acc = val_acc
                            # Save all test metrics at this optimal validation epoch
                            fold_best_metrics = {k: v for k, v in data.items() if k.startswith('test_')}
                            fold_best_metrics['optimal_epoch'] = epoch
                            fold_best_metrics['fold'] = fold_name

                    except json.JSONDecodeError:
                        continue
                        
        except IOError as e:
            print(f"Error reading {log_path}: {e}")
            continue
            
        if fold_best_metrics:
            best_test_metrics.append(fold_best_metrics)

    return epochs_data, best_test_metrics


def plot_cv_curves(epochs_data, output_dir):
    """
    Plots the mean and standard deviation of metrics across folds.
    """
    if not epochs_data:
        return

    # Extract the maximum epoch reached by any fold
    all_epochs = sorted(list(epochs_data['epoch'].keys()))

    plot_configurations = [
        {
            "title": "Loss Curves (10-Fold CV)",
            "ylabel": "Loss",
            "filename": "loss_curves_cv.png",
            "metrics": {
                "train_loss": "Train Loss",
                "val_loss": "Validation Loss",
                "test_loss": "Test Loss"
            }
        },
        {
            "title": "Accuracy Curves (10-Fold CV)",
            "ylabel": "Balanced Accuracy",
            "filename": "accuracy_curves_cv.png",
            "metrics": {
                "train_class_acc": "Train Accuracy",
                "val_balanced_accuracy": "Validation Balanced Accuracy",
                "test_balanced_accuracy": "Test Balanced Accuracy"
            }
        },
        {
            "title": "ROC AUC Curves (10-Fold CV)",
            "ylabel": "ROC AUC",
            "filename": "roc_auc_curves_cv.png",
            "metrics": {
                "val_roc_auc": "Validation ROC AUC",
                "test_roc_auc": "Test ROC AUC"
            }
        },
        {
            "title": "PR AUC Curves (10-Fold CV)",
            "ylabel": "PR AUC",
            "filename": "pr_auc_curves_cv.png",
            "metrics": {
                "val_pr_auc": "Validation PR AUC",
                "test_pr_auc": "Test PR AUC"
            }
        }
    ]

    split_colors = {
        "train": "#A3D5FF",
        "val":   "#FFCB8D",
        "test":  "#FF7886",
    }

    split_legend_names = {
        "train": "Train (Mean ± Std)",
        "val": "Validation (Mean ± Std)",
        "test": "Test (Mean ± Std)",
    }

    os.makedirs(output_dir, exist_ok=True)

    for config in plot_configurations:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        for key, display_name in config["metrics"].items():
            if key not in epochs_data:
                continue

            means = []
            stds = []
            valid_epochs = []

            for epoch in all_epochs:
                values = epochs_data[key].get(epoch, [])
                # Only compute stats if we have data for this epoch
                if values and not np.all(np.isnan(values)):
                    means.append(np.nanmean(values))
                    stds.append(np.nanstd(values))
                    valid_epochs.append(epoch)

            if not means:
                continue

            means = np.array(means)
            stds = np.array(stds)

            split = key.split('_')[0]  # 'train', 'val', or 'test'
            color = split_colors.get(split, 'grey')
            label = split_legend_names.get(split, split)

            # Plot the mean line
            ax.plot(
                valid_epochs, means,
                marker='o', linestyle='-', markersize=4,
                color=color,
                label=label
            )

            # Plot the standard deviation shading
            ax.fill_between(
                valid_epochs, means - stds, means + stds,
                color=color, alpha=0.2
            )

        ax.set_title(config["title"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(config["ylabel"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="0.85", linewidth=0.8)
        ax.grid(False, axis="x")

        if all_epochs:
            start = int(np.min(all_epochs))
            end = int(np.max(all_epochs))
            ax.set_xticks(np.arange(start, end + 2, 5))

        ax.legend()

        plot_filename = os.path.join(output_dir, config["filename"])
        plt.savefig(plot_filename, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"Saved plot: {plot_filename}")


def save_cv_summary(best_test_metrics, output_dir):
    """
    Computes the final Mean and Std across all folds and saves to a text file.
    """
    if not best_test_metrics:
        return

    results_path = os.path.join(output_dir, 'cv_results_summary.txt')
    
    # Extract keys that start with 'test_'
    metric_keys = [k for k in best_test_metrics[0].keys() if k.startswith('test_')]
    
    with open(results_path, 'w') as f:
        f.write("10-Fold Cross-Validation Summary\n")
        f.write("Method: Model selected per fold based on highest Validation Balanced Accuracy.\n")
        f.write("="*70 + "\n\n")
        
        # 1. Write the aggregated summary
        f.write("### FINAL AGGREGATED METRICS (MEAN ± STD) ###\n")
        for key in metric_keys:
            values = [fold_data[key] for fold_data in best_test_metrics if key in fold_data]
            mean_val = np.mean(values)
            std_val = np.std(values)
            f.write(f"{key}: {mean_val:.4f} ± {std_val:.4f}\n")
            
        f.write("\n" + "="*70 + "\n\n")
        
        # 2. Write the breakdown per fold for transparency
        f.write("### BREAKDOWN PER FOLD ###\n")
        for fold_data in best_test_metrics:
            f.write(f"--- {fold_data['fold']} (Optimal Epoch: {fold_data['optimal_epoch']}) ---\n")
            for key in metric_keys:
                if key in fold_data:
                    f.write(f"  {key}: {fold_data[key]:.4f}\n")
            f.write("\n")

    print(f"Saved CV numerical summary to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='LaBraM 10-Fold CV evaluation and plotting script.')
    parser.add_argument(
        '--cv_dir', required=True, type=str,
        help='Path to the main CV directory containing fold_0, fold_1, etc.'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.cv_dir):
        print(f"Error: Directory not found at {args.cv_dir}")
        return

    print("Parsing cross-validation logs...")
    epochs_data, best_test_metrics = parse_cv_logs(args.cv_dir)

    if epochs_data:
        print("\nPlotting cross-validation training curves...")
        plot_cv_curves(epochs_data, args.cv_dir)
        
        print("\nSaving numerical summary...")
        save_cv_summary(best_test_metrics, args.cv_dir)
    else:
        print("No valid log data found to evaluate.")

if __name__ == '__main__':
    main()