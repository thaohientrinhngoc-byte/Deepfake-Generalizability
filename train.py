"""
Training script for Spatiotemporal Deepfake Detectors.

Supports Paper Models (Table I):
  - M_CNN_F ($M_{CNN-F}$): Frozen CNN Backbone + GAP
  - M_CNN_T ($M_{CNN-T}$): Tuned CNN Backbone (Blocks 7, 8) + GAP
  - M_CL_F  ($M_{CL-F}$) : Frozen CNN Backbone + 3-Layer LSTM
  - M_CL_T  ($M_{CL-T}$) : Tuned CNN Backbone (Blocks 7, 8) + 3-Layer LSTM

Supports single-seed and multi-seed (42, 36, 75) training protocols as described in Section 4.1.
"""

import os
import gc
import json
import shutil
import random
import logging
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from models import SpatiotemporalDeepfakeDetector
from utils import DeepfakeDataset, DynamicBalancedSampler, collate_fn, evaluate_predictions


def get_logger(exp_dir: str) -> logging.Logger:
    logger_name = f"train_logger_{os.path.abspath(exp_dir)}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(exp_dir, "training.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def prepare_batch(x: torch.Tensor, y: torch.Tensor, device: torch.device):
    x = x.to(device, non_blocking=True).float().div_(255.0).contiguous()
    y = y.to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def run_evaluation(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for x, y in loader:
        x, y = prepare_batch(x, y, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits, y)

        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs.tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_scores = np.array(all_preds)
    metrics = evaluate_predictions(y_true, y_scores)
    metrics["Loss"] = total_loss / max(n_batches, 1)
    metrics["y_true"] = y_true
    metrics["y_scores"] = y_scores
    return metrics


def train_experiment(
    model_type: str,
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = config.BATCH_SIZE,
    base_lr: float = config.DEFAULT_LR,
    seed: int = config.RANDOM_SEED,
    use_seed_subdir: bool = False,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_folder = model_type.lower().replace("-", "_")
    base_exp_dir = os.path.join(output_dir, model_folder)
    exp_dir = os.path.join(base_exp_dir, f"seed_{seed}") if use_seed_subdir else base_exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    logger = get_logger(exp_dir)

    logger.info(f"=== Starting Training: Model {model_type} | Seed {seed} ===")
    logger.info(f"Device: {device} | Output Directory: {exp_dir}")

    # Load Data
    train_ds = DeepfakeDataset(data_dir, split="train", is_train=True)
    val_ds   = DeepfakeDataset(data_dir, split="val",   is_train=False)
    test_ds  = DeepfakeDataset(data_dir, split="test",  is_train=False)

    train_labels = [s[1] for s in train_ds.samples]
    train_loader = DataLoader(
        train_ds,
        batch_sampler=DynamicBalancedSampler(train_labels, batch_size),
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = SpatiotemporalDeepfakeDetector(model_type=model_type).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    best_val_auc  = -1.0
    no_improve    = 0
    history_path  = os.path.join(exp_dir, "history.jsonl")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss, train_batches = 0.0, 0
        lr = optimizer.param_groups[0]["lr"]

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = prepare_batch(x, y, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / max(train_batches, 1)
        val_metrics = run_evaluation(model, val_loader, criterion, device)
        scheduler.step(val_metrics["Loss"])

        logger.info(
            f"Epoch [{epoch}/{epochs}] LR: {lr:.1e} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['Loss']:.4f} | Val AUC: {val_metrics['AUC']:.4f} | Val EER: {val_metrics['EER']:.4f}"
        )

        improved = False
        if val_metrics["Loss"] < best_val_loss:
            best_val_loss = val_metrics["Loss"]
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_loss_model.pth"))
            improved = True

        if val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_auc_model.pth"))
            improved = True

        with open(history_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch, "lr": lr,
                "train_loss": avg_train_loss,
                "val_loss": val_metrics["Loss"],
                "val_acc": val_metrics["ACC"],
                "val_auc": val_metrics["AUC"],
                "val_eer": val_metrics["EER"],
            }) + "\n")

        no_improve = 0 if improved else no_improve + 1
        if no_improve >= 5:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    # Final Evaluation on Test Split
    logger.info("Executing Final Test evaluation with Best AUC Checkpoint...")
    best_model_path = os.path.join(exp_dir, "best_auc_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        if use_seed_subdir and not os.path.exists(os.path.join(base_exp_dir, "best_auc_model.pth")):
            shutil.copy2(best_model_path, os.path.join(base_exp_dir, "best_auc_model.pth"))

    test_metrics = run_evaluation(model, test_loader, criterion, device)
    logger.info(
        f"--- TEST RESULTS (Seed {seed}) --- Loss: {test_metrics['Loss']:.4f} | ACC: {test_metrics['ACC']:.4f} | "
        f"AUC: {test_metrics['AUC']:.4f} | EER: {test_metrics['EER']:.4f}"
    )

    clean_metrics = {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
    with open(os.path.join(exp_dir, "test_results.json"), "w") as f:
        json.dump(clean_metrics, f, indent=2)

    logger.info("Training experiment finished successfully.")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return clean_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spatiotemporal Deepfake Detector")
    parser.add_argument("--model", type=str, required=True, choices=["M_CNN_F", "M_CNN_T", "M_CL_F", "M_CL_T"], help="Model configuration name from Paper Table I")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed dataset directory")
    parser.add_argument("--output_dir", type=str, default="./models_checkpoints", help="Directory to save model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Single random seed (default: 42)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Multiple random seeds (e.g. --seeds 42 36 75)")

    args = parser.parse_args()

    if args.seeds and len(args.seeds) > 0:
        all_seed_metrics = []
        for s in args.seeds:
            m = train_experiment(
                model_type=args.model,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                base_lr=args.lr,
                seed=s,
                use_seed_subdir=True,
            )
            all_seed_metrics.append(m)

        # Report Mean ± Std across seeds (in %)
        print(f"\n=== MULTI-SEED SUMMARY ({args.model}, Seeds={args.seeds}) ===")
        for key in ["AUC", "ACC", "F1", "Recall_Real", "Recall_Fake", "EER"]:
            vals = [m[key] * 100.0 for m in all_seed_metrics if key in m]
            if vals:
                ddof = 1 if len(vals) > 1 else 0
                print(f"  {key:12s}: {np.mean(vals):.2f} ± {np.std(vals, ddof=ddof):.2f} (%)")
    else:
        train_experiment(
            model_type=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_lr=args.lr,
            seed=args.seed,
            use_seed_subdir=False,
        )
