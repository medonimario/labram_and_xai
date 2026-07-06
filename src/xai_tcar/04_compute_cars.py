import numpy as np
import os
import pickle
import json
import argparse
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dotenv import load_dotenv

# --- 1. Compute CAR (Kernel SVM with CV) ---
def compute_car_svm_cv(X_train_val, y_train_val, X_test, y_test):
    """
    Trains a non-linear Support Vector Classifier (RBF kernel) using Grid Search 
    and Stratified Cross-Validation to distinguish Concept from Random.
    Returns: Fitted best SVC model, Best Parameters, Train Metrics, Test Metrics.
    """
    # Define the hyperparameter grid to search over
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
    }
    
    # 5-fold Stratified Cross-Validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Base Model (RBF kernel is required for latent space isometry invariance)
    base_clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    
    # Grid Search Initialization
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1 # Use all available CPU cores
    )
    
    # Fit the grid search (runs the CV loop internally)
    grid_search.fit(X_train_val, y_train_val)
    
    # Extract the best model and parameters
    best_clf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    # Metrics on the entire train_val set
    acc_train = best_clf.score(X_train_val, y_train_val)
    
    # Unbiased Metrics on the completely unseen holdout test set
    acc_test = best_clf.score(X_test, y_test)
    
    # For AUC we use the decision function (distance to the non-linear boundary)
    scores_test = best_clf.decision_function(X_test)
    try:
        auc_test = roc_auc_score(y_test, scores_test)
        # Handle inverted direction for AUC if concept was learned backwards
        if auc_test < 0.5:
            auc_test = 1.0 - auc_test
    except ValueError:
        auc_test = 0.5
        
    return best_clf, best_params, {'acc': acc_train}, {'acc': acc_test, 'auc': auc_test, 'cv_mean_acc': best_cv_score}


def main(args):
    load_dotenv()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    target_layers = [int(l) for l in args.target_layers.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Concept Activations
    # We load these once as they stay constant across all random runs
    print(f"Loading Concept Activations for: {args.concept_name}")
    concept_acts_by_layer = {}
    for layer_id in target_layers:
        p = os.path.join(args.concept_activation_dir, f"{args.concept_name}_layer_{layer_id}.pkl")
        if os.path.exists(p):
            with open(p, 'rb') as f:
                concept_acts_by_layer[layer_id] = pickle.load(f)
        else:
            print(f"Warning: Concept activations missing for layer {layer_id}")

    # Initialize Storage
    # Structure: results_store[layer_id] = [{'model': SVC_object, 'metrics': {...}, 'best_params': {...}}, ...]
    results_store = {l: [] for l in target_layers}

    print(f"\n--- Starting CAR Computation with Cross-Validation ({args.num_runs} Random Runs) ---")
    
    for run_i in tqdm(range(args.num_runs), desc="Runs"):
        random_name = f"random_run_{run_i}"
        
        for layer_id in target_layers:
            if layer_id not in concept_acts_by_layer: 
                continue
            
            concept_acts = concept_acts_by_layer[layer_id]
            
            # Load Random Acts for THIS run and THIS layer
            rand_path = os.path.join(args.random_activation_dir, f"{random_name}_layer_{layer_id}.pkl")
            if not os.path.exists(rand_path):
                continue # Skip if file missing
                
            with open(rand_path, 'rb') as f:
                random_acts = pickle.load(f)

            # --- Prepare Data ---
            # X: Concatenate Concept (1) and Random (0)
            X = np.concatenate((concept_acts, random_acts), axis=0)
            y = np.concatenate((np.ones(len(concept_acts)), np.zeros(len(random_acts))))
            
            # Split Data: 80% for CV Tuning (train_val), 20% strictly for holdout validation (test)
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )

            # --- Compute CAR with Grid Search CV ---
            car_model, best_params, stats_train, stats_test = compute_car_svm_cv(
                X_train_val, y_train_val, X_test, y_test
            )
            
            results_store[layer_id].append({
                'run_id': run_i,
                'model': car_model,
                'best_params': best_params,
                'metrics': {
                    **{f'train_{k}': v for k, v in stats_train.items()},
                    **{f'test_{k}': v for k, v in stats_test.items()}
                }
            })

    # --- Save Results ---
    print("\nSaving CAR models and metrics...")
    
    metrics_summary = {}

    for layer_id in target_layers:
        if not results_store[layer_id]:
            continue
            
        layer_models = []
        layer_metrics = []
        
        # Unpack data
        for res in results_store[layer_id]:
            layer_models.append(res['model'])
            # Attach best params directly to the run's metrics for easier JSON inspection
            run_summary = res['metrics'].copy()
            run_summary['best_params'] = res['best_params']
            layer_metrics.append(run_summary)

        # 1. Save Models (Pickle full SVC objects)
        model_path = os.path.join(args.output_dir, f"cars_{args.concept_name}_layer_{layer_id}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(layer_models, f)
            
        # 2. Collect Metrics for JSON
        metrics_summary[layer_id] = layer_metrics

    # Save Metrics (JSON)
    json_path = os.path.join(args.output_dir, f"metrics_{args.concept_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"Models saved to: {args.output_dir}/cars_*.pkl")
    print(f"Metrics saved to: {json_path}")
    
    # --- Quick Report ---
    print("\n--- Quick Metric Summary ---")
    for l, metrics in metrics_summary.items():
        mean_test_acc = np.mean([m['test_acc'] for m in metrics])
        mean_test_auc = np.mean([m['test_auc'] for m in metrics])
        mean_cv_acc = np.mean([m['test_cv_mean_acc'] for m in metrics])
        print(f"Layer {l}: Holdout Accuracy = {mean_test_acc:.3f}")
        print(f"         Holdout AUC      = {mean_test_auc:.3f}")
        print(f"         Internal CV Acc  = {mean_cv_acc:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Concept Activation Regions (CARs) using CV and RBF SVMs")
    
    parser.add_argument("--concept_activation_dir", type=str, required=True)
    parser.add_argument("--random_activation_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    parser.add_argument("--concept_name", type=str, default="concept_set")
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)