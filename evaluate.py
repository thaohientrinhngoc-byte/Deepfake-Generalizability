"""
Cross-Dataset Matrix Evaluation & DGS Score Computation.

Evaluates 4 Spatiotemporal Models across 4 Benchmark Datasets:
  - FaceForensics++ (FF++)
  - DeepFake Detection Challenge (DFDC)
  - Celeb-DF (v2)
  - WildDeepfake (Wild)

Supports both single-seed checkpoints and multi-seed (42, 36, 75) evaluation
to reproduce the Mean ± Std (%) tables and DGS_AUC scores in Tables III–VI of the paper.

Outputs:
  - Complete metrics CSV & Pivot Matrices (AUC, ACC, EER, F1, DGS)
  - Summary Table (Mean ± Std %) matching Paper Tables III–VI
  - Confusion Matrix plots per model-dataset combination
  - Combined ROC Curves per dataset
"""

import os
import gc
import json
import argparse
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve

import config
from models import SpatiotemporalDeepfakeDetector
from utils import DeepfakeDataset, collate_fn, evaluate_predictions, compute_dgs_score


def process_and_plot_cm(y_true: np.ndarray, y_pred_labels: np.ndarray, filepath: str, title: str):
    cm = confusion_matrix(y_true, y_pred_labels)
    group_counts = [f"{value:0.0f}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax, xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    fig.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close(fig)


@torch.no_grad()
def evaluate_model_on_dataset(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True).float().div_(255.0).contiguous()
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs.tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_scores = np.array(all_preds)
    metrics = evaluate_predictions(y_true, y_scores)
    metrics["y_true"] = y_true
    metrics["y_scores"] = y_scores
    return metrics


def find_checkpoints(base_dir: str, model_name: str, train_set: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Finds checkpoint files under candidate paths.
    Returns a list of (seed_label, checkpoint_path) tuples.
    Supports multi-seed subdirectories (e.g. seed_42, seed_36, seed_75) or single checkpoint.
    """
    candidate_names = ["best_auc_model.pth", "best_loss_model.pth"]
    model_aliases = [model_name, model_name.lower(), model_name.lower().replace("-", "_")]

    subdirs = []
    if train_set:
        train_aliases = [train_set, train_set.lower(), train_set.upper(), train_set.replace("_", "-")]
        for t in train_aliases:
            for m in model_aliases:
                subdirs.append(os.path.join(base_dir, t, m))
    for m in model_aliases:
        subdirs.append(os.path.join(base_dir, m))
    subdirs.append(base_dir)

    # First check for multi-seed subdirectories (seed_42, seed_36, seed_75)
    for s in subdirs:
        if not os.path.isdir(s):
            continue
        found_seeds = []
        for seed_val in config.EXPERIMENT_SEEDS:
            seed_dir = os.path.join(s, f"seed_{seed_val}")
            for c in candidate_names:
                p = os.path.join(seed_dir, c)
                if os.path.isfile(p):
                    found_seeds.append((str(seed_val), p))
                    break
        if found_seeds:
            return found_seeds

    # Fallback to single checkpoint
    for s in subdirs:
        for c in candidate_names:
            p = os.path.join(s, c)
            if os.path.isfile(p):
                return [(str(config.RANDOM_SEED), p)]
    return []


def match_internal_dataset(train_name: str, test_names: List[str]) -> Optional[str]:
    """Matches training dataset name against available test dataset keys."""
    t_norm = train_name.lower().replace("-", "").replace("_", "").replace("+", "").replace(" ", "")
    for d in test_names:
        d_norm = d.lower().replace("-", "").replace("_", "").replace("+", "").replace(" ", "")
        if t_norm == d_norm or t_norm in d_norm or d_norm in t_norm:
            return d
    return None


def fmt_mean_std_pct(values: List[float]) -> str:
    """Formats a list of decimal metric values [0, 1] as 'XX.XX ± YY.YY' in percentage."""
    pct_vals = [v * 100.0 for v in values if not np.isnan(v)]
    if not pct_vals:
        return "N/A"
    if len(pct_vals) == 1:
        return f"{pct_vals[0]:.2f}"
    mean_v = float(np.mean(pct_vals))
    std_v = float(np.std(pct_vals, ddof=1))
    return f"{mean_v:.2f} ± {std_v:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Deepfake Detector Evaluation")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Directory containing trained model weights")
    parser.add_argument("--datasets_config", type=str, default="datasets.json", help="JSON file mapping Dataset Names to directory paths")
    parser.add_argument("--train_dataset", type=str, default=None, help="Name of training dataset (e.g. DFDC, CELEB_DF, FF++) to compute DGS")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save evaluation artifacts")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Evaluation batch size")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.datasets_config):
        raise FileNotFoundError(f"Datasets config file not found at '{args.datasets_config}'. Please create one with dataset paths.")

    with open(args.datasets_config, "r") as f:
        dataset_paths = json.load(f)

    models_to_eval = ["M_CNN_F", "M_CNN_T", "M_CL_F", "M_CL_T"]

    # Load Data Loaders
    loaders_dict = {}
    for d_name, d_path in dataset_paths.items():
        if not os.path.exists(d_path):
            print(f"Notice: Dataset path for '{d_name}' not found at '{d_path}', skipping.")
            continue
        ds = DeepfakeDataset(d_path, split="test", is_train=False)
        if len(ds) > 0:
            loaders_dict[d_name] = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True
            )
            print(f"Loaded test dataset '{d_name}': {len(ds)} samples.")

    if not loaders_dict:
        print("Error: No test datasets could be loaded from datasets_config.")
        return

    train_datasets = [args.train_dataset] if args.train_dataset else [None]

    raw_results = []
    summary_rows = []
    roc_data = {}

    for train_set in train_datasets:
        for model_name in models_to_eval:
            seed_ckpts = find_checkpoints(args.checkpoints_dir, model_name, train_set)
            if not seed_ckpts:
                print(f"Warning: Checkpoint not found for model '{model_name}' (Train set: {train_set}) under {args.checkpoints_dir}")
                continue

            print(f"\nEvaluating Model: {model_name} (Trained on: {train_set or 'Unspecified'}, Seeds: {[s for s, _ in seed_ckpts]})...")
            per_dataset_seed_metrics: Dict[str, List[dict]] = {d_name: [] for d_name in loaders_dict}

            for seed_idx, (seed_label, weights_path) in enumerate(seed_ckpts):
                print(f"  [Seed {seed_label}] Checkpoint: {weights_path}")
                model = SpatiotemporalDeepfakeDetector(model_type=model_name)
                model.load_state_dict(torch.load(weights_path, map_location=device))
                model = model.to(device)

                for d_name, loader in loaders_dict.items():
                    metrics = evaluate_model_on_dataset(model, loader, device)
                    per_dataset_seed_metrics[d_name].append(metrics)

                    raw_results.append({
                        "Training Set": train_set if train_set else "N/A",
                        "Model": model_name,
                        "Seed": seed_label,
                        "Testing Set": d_name,
                        "AUC": metrics["AUC"],
                        "ACC": metrics["ACC"],
                        "EER": metrics["EER"],
                        "F1-Score": metrics["F1"],
                        "Recall (Real)": metrics["Recall_Real"],
                        "Recall (Fake)": metrics["Recall_Fake"],
                        "Precision (Real)": metrics["Precision_Real"],
                        "Precision (Fake)": metrics["Precision_Fake"],
                    })

                    # Plot Confusion Matrix & ROC for primary seed
                    if seed_idx == 0:
                        prefix = f"{train_set}_" if train_set else ""
                        cm_filename = os.path.join(args.output_dir, f"cm_{prefix}{model_name.lower()}_{d_name.lower()}.png")
                        process_and_plot_cm(metrics["y_true"], (metrics["y_scores"] >= 0.5).astype(int), cm_filename, f"{model_name} on {d_name}")

                        roc_key = f"{train_set}_{d_name}" if train_set else d_name
                        if roc_key not in roc_data:
                            roc_data[roc_key] = []
                        roc_data[roc_key].append({
                            "Model": model_name,
                            "y_true": metrics["y_true"],
                            "y_scores": metrics["y_scores"],
                            "auc": metrics["AUC"],
                        })

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Compute mean AUCs per dataset across seeds to compute DGS_AUC (Equation 1)
            mean_aucs = {
                d_name: float(np.mean([m["AUC"] for m in m_list]))
                for d_name, m_list in per_dataset_seed_metrics.items() if m_list
            }
            internal_key = match_internal_dataset(train_set, list(loaders_dict.keys())) if train_set else None
            dgs_pct_str = ""
            if internal_key and internal_key in mean_aucs:
                a_internal = mean_aucs[internal_key]
                cross_aucs = [v for k, v in mean_aucs.items() if k != internal_key and not np.isnan(v)]
                if cross_aucs:
                    dgs_val = compute_dgs_score(a_internal, cross_aucs)
                    dgs_pct_str = f"{dgs_val * 100.0:.2f}"
                    print(f"==> Dataset Generalizability Score (DGS_AUC) for {train_set} on {model_name}: {dgs_pct_str}%")

            for d_name, m_list in per_dataset_seed_metrics.items():
                summary_rows.append({
                    "Training Set": train_set if train_set else "N/A",
                    "Model": model_name,
                    "Testing Set": d_name,
                    "AUC (%)": fmt_mean_std_pct([m["AUC"] for m in m_list]),
                    "ACC (%)": fmt_mean_std_pct([m["ACC"] for m in m_list]),
                    "F1-Score (%)": fmt_mean_std_pct([m["F1"] for m in m_list]),
                    "Recall Real (%)": fmt_mean_std_pct([m["Recall_Real"] for m in m_list]),
                    "Recall Fake (%)": fmt_mean_std_pct([m["Recall_Fake"] for m in m_list]),
                    "EER (%)": fmt_mean_std_pct([m["EER"] for m in m_list]),
                    "DGS_AUC (%)": dgs_pct_str if (internal_key and d_name == internal_key) else "",
                })

    if not raw_results:
        print("No valid evaluation results were computed.")
        return

    df_raw = pd.DataFrame(raw_results)
    df_raw.to_csv(os.path.join(args.output_dir, "complete_testing_report.csv"), index=False)

    # Mean AUC Pivot Matrix
    auc_matrix = df_raw.groupby(["Model", "Testing Set"])["AUC"].mean().unstack()
    print("\n" + "=" * 65)
    print(" MEAN AUC PERFORMANCE MATRIX ")
    print("=" * 65)
    print(auc_matrix.to_string())
    auc_matrix.to_csv(os.path.join(args.output_dir, "matrix_auc.csv"))

    # Summary table formatted matching Paper Tables III–VI
    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 85)
    print(" EXPERIMENTAL SUMMARY REPORT (Unit: %, Mean ± Std across Seeds) ")
    print("=" * 85)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(os.path.join(args.output_dir, "summary_table.csv"), index=False)

    # Plot ROC Curves
    for d_name, models_list in roc_data.items():
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")

        for m_info in models_list:
            fpr, tpr, _ = roc_curve(m_info["y_true"], m_info["y_scores"])
            ax.plot(fpr, tpr, lw=2, label=f"{m_info['Model']} (AUC = {m_info['auc']:.4f})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title(f"ROC Curves on Dataset: {d_name}", fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)

        roc_path = os.path.join(args.output_dir, f"roc_curve_{d_name.lower()}.png")
        fig.savefig(roc_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    print(f"\nEvaluation completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
