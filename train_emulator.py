from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from emulator_training.data import (
    PreparedData,
    apply_standard_scaler,
    fit_standard_scaler,
    fit_target_transforms,
    inverse_transform_targets,
    prepare_dataset,
    regression_metrics,
    transform_targets,
)
from emulator_training.model import build_model


def log_training(message: str) -> None:
    print(f"[TRAIN] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline emulator for coarse-grained CASS LES moments.")
    parser.add_argument("--config", default="configs/baseline_mlp.yaml", help="Path to the YAML config file.")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--device", default="auto", help="`auto`, `cpu`, or a torch device string.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for the number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override for the batch size.")
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional limit used for smoke tests and rapid iteration.",
    )
    parser.add_argument("--disable-plots", action="store_true", help="Skip diagnostic plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)

    output_dir = resolve_output_dir(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["training"]["seed"]))
    device = resolve_device(args.device)

    data = prepare_dataset(config, max_samples_per_split=args.max_samples_per_split)
    log_training("Fitting feature and target scalers")
    scalers = fit_scalers(data, config)
    log_training("Building standardized tensors and dataloaders")
    loaders = build_dataloaders(data, scalers, config, device)
    log_training(f"Building model on device={device}")

    model = build_model(
        model_config=config["model"],
        input_dim=len(data.feature_names),
        output_dim=len(data.target_names),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    log_training("Starting optimization")
    history, best_state = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=int(config["training"]["epochs"]),
        patience=int(config["training"]["early_stopping_patience"]),
        device=device,
    )
    model.load_state_dict(best_state)

    metrics, predictions = evaluate_all_splits(
        model,
        data,
        scalers,
        device,
        batch_size=int(config["training"]["batch_size"]),
    )
    profile_diagnostics = build_profile_diagnostics(data, predictions)
    importance_results = compute_permutation_importance(
        model=model,
        data=data,
        scalers=scalers,
        metrics=metrics,
        device=device,
        batch_size=int(config["training"]["batch_size"]),
        importance_config=config.get("importance", {}),
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "feature_scaler": serialize_scaler(scalers["feature_scaler"]),
        "target_transforms": scalers["target_transforms"],
        "metadata": data.metadata,
        "history": history,
        "metrics": metrics,
        "importance": importance_results,
    }
    torch.save(checkpoint, output_dir / "model_checkpoint.pt")

    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "metadata.json", data.metadata)
    save_json(output_dir / "normalization.json", {
        "feature_scaler": serialize_scaler(scalers["feature_scaler"]),
        "target_transforms": scalers["target_transforms"],
    })
    save_json(output_dir / "feature_importance.json", importance_results)
    write_profile_diagnostics_netcdf(output_dir / "vertical_profile_diagnostics.nc", profile_diagnostics, data)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    if not args.disable_plots:
        create_plots(output_dir, history, metrics, profile_diagnostics, importance_results, data.target_names)

    print(f"Training complete. Artifacts written to {output_dir}")
    for split_name, split_metrics in metrics.items():
        print(f"[{split_name}]")
        for target_name, target_metrics in split_metrics.items():
            summary = ", ".join(f"{key}={value:.5f}" for key, value in target_metrics.items())
            print(f"  {target_name}: {summary}")


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)


def resolve_output_dir(config: dict[str, Any], cli_output_dir: str | None) -> Path:
    if cli_output_dir is not None:
        return Path(cli_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(config["training"]["output_root"])
    return base / f"{config['experiment_name']}_{timestamp}"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_scalers(data: PreparedData, config: dict[str, Any]) -> dict[str, Any]:
    log_training(f"Computing feature scaler from training features with shape={data.train.features.shape}")
    feature_scaler = fit_standard_scaler(data.train.features)
    log_training(f"Computing target transforms from training targets with shape={data.train.targets_raw.shape}")
    target_transforms = fit_target_transforms(
        raw_targets=data.train.targets_raw,
        target_names=data.target_names,
        transform_config=config["data"]["target_transforms"],
    )
    log_training("Finished scaler fitting")
    return {"feature_scaler": feature_scaler, "target_transforms": target_transforms}


def build_dataloaders(
    data: PreparedData,
    scalers: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, DataLoader]:
    training_cfg = config["training"]
    batch_size = int(config["training"]["batch_size"])
    pin_memory = device.type == "cuda"
    shuffle_mode = str(training_cfg.get("train_shuffle_mode", "per_epoch"))
    shuffle_seed = int(training_cfg.get("train_shuffle_seed", training_cfg.get("seed", 7)))

    if shuffle_mode not in {"per_epoch", "once", "none"}:
        raise ValueError(f"Unsupported train_shuffle_mode {shuffle_mode!r}.")

    loaders = {}
    for split_name in ("train", "val", "test"):
        split = getattr(data, split_name)
        log_training(
            f"Preparing split={split_name} features_shape={split.features.shape} targets_shape={split.targets_raw.shape}"
        )
        features = apply_standard_scaler(split.features, scalers["feature_scaler"])
        targets = transform_targets(split.targets_raw, scalers["target_transforms"])

        shuffle = split_name == "train" and shuffle_mode == "per_epoch"
        if split_name == "train" and shuffle_mode == "once":
            log_training(f"Applying one-time training shuffle with seed={shuffle_seed}")
            order = np.random.default_rng(shuffle_seed).permutation(len(features))
            features = features[order]
            targets = targets[order]

        dataset = TensorDataset(
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(targets.astype(np.float32)),
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=pin_memory,
        )
        log_training(
            f"Finished split={split_name} tensor preparation with samples={len(dataset)} "
            f"and batches={len(loaders[split_name])}"
        )
    return loaders


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device, training=True)
        val_loss = run_epoch(model, val_loader, optimizer, loss_fn, device, training=False)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        print(f"Epoch {epoch + 1:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    return history, best_state


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    training: bool,
) -> float:
    total_loss = 0.0
    total_count = 0

    if training:
        model.train()
    else:
        model.eval()

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(training):
            prediction = model(features)
            loss = loss_fn(prediction, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate_all_splits(
    model: nn.Module,
    data: PreparedData,
    scalers: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, np.ndarray]]:
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    predictions: dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test"):
        split = getattr(data, split_name)
        prediction = predict_split(model, split.features, scalers, device, batch_size)
        predictions[split_name] = prediction
        split_metrics = {}
        for idx, target_name in enumerate(data.target_names):
            split_metrics[target_name] = regression_metrics(split.targets_raw[..., idx], prediction[..., idx])
        metrics[split_name] = split_metrics
    return metrics, predictions


def predict_split(
    model: nn.Module,
    features: np.ndarray,
    scalers: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    normalized = apply_standard_scaler(features, scalers["feature_scaler"])
    dataset = TensorDataset(torch.from_numpy(normalized.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    preds = []
    model.eval()
    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(device)
            batch_pred = model(batch_features).cpu().numpy()
            preds.append(batch_pred)
    normalized_pred = np.concatenate(preds, axis=0)
    return inverse_transform_targets(normalized_pred, scalers["target_transforms"])


def serialize_scaler(scaler: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {name: values.astype(float).tolist() for name, values in scaler.items()}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def create_plots(
    output_dir: Path,
    history: dict[str, list[float]],
    metrics: dict[str, dict[str, dict[str, float]]],
    profile_diagnostics: dict[str, dict[str, np.ndarray]],
    importance_results: dict[str, Any],
    target_names: list[str],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib.pyplot as plt

    plot_dpi = 120

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=plot_dpi)
    plt.close(fig)

    fig, axes = plt.subplots(len(target_names), 1, figsize=(5.6, 2.4 * len(target_names)))
    if len(target_names) == 1:
        axes = [axes]

    for ax, target_name in zip(axes, target_names):
        split_metrics = metrics["val"][target_name]
        text = "\n".join(
            [
                f"RMSE: {split_metrics['rmse']:.4f}",
                f"MAE:  {split_metrics['mae']:.4f}",
                f"Bias: {split_metrics['bias']:.4f}",
                f"Corr: {split_metrics['corr']:.4f}",
                f"R2:   {split_metrics['r2']:.4f}",
            ]
        )
        ax.axis("off")
        ax.text(0.03, 0.95, f"Validation metrics for {target_name}", va="top", fontsize=10)
        ax.text(0.03, 0.72, text, va="top", family="monospace", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "validation_metrics.png", dpi=plot_dpi)
    plt.close(fig)

    plotted_splits = [split_name for split_name in ("train", "val", "test") if split_name in profile_diagnostics]
    fig, axes = plt.subplots(
        len(target_names),
        len(plotted_splits),
        figsize=(3.5 * len(plotted_splits), 2.8 * len(target_names)),
        sharey=True,
    )
    if len(target_names) == 1 and len(plotted_splits) == 1:
        axes = np.array([[axes]])
    elif len(target_names) == 1:
        axes = np.array([axes])
    elif len(plotted_splits) == 1:
        axes = np.array([[ax] for ax in axes])

    for row_idx, target_name in enumerate(target_names):
        for col_idx, split_name in enumerate(plotted_splits):
            ax = axes[row_idx, col_idx]
            diag = profile_diagnostics[split_name]
            target_idx = target_names.index(target_name)
            height = diag["height_m"]
            truth_mean = diag["truth_mean"][:, target_idx]
            prediction_mean = diag["prediction_mean"][:, target_idx]
            bias_mean = diag["bias_mean"][:, target_idx]

            ax.plot(truth_mean, height, label="Target", color="black", linewidth=2.0)
            ax.plot(prediction_mean, height, label="ML", color="#d95f02", linewidth=2.0)
            ax.plot(bias_mean, height, label="ML - target", color="#1b9e77", linestyle="--", linewidth=1.5)
            ax.set_xlabel(target_name)
            if col_idx == 0:
                ax.set_ylabel("Height (m)")
            ax.set_title(f"{split_name.capitalize()} mean profile", fontsize=10)
            ax.grid(True, alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "mean_vertical_profiles_train_val_test.png", dpi=plot_dpi)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(target_names),
        len(plotted_splits),
        figsize=(3.5 * len(plotted_splits), 2.8 * len(target_names)),
        sharey=True,
    )
    if len(target_names) == 1 and len(plotted_splits) == 1:
        axes = np.array([[axes]])
    elif len(target_names) == 1:
        axes = np.array([axes])
    elif len(plotted_splits) == 1:
        axes = np.array([[ax] for ax in axes])

    for row_idx, target_name in enumerate(target_names):
        for col_idx, split_name in enumerate(plotted_splits):
            ax = axes[row_idx, col_idx]
            diag = profile_diagnostics[split_name]
            target_idx = target_names.index(target_name)
            height = diag["height_m"]

            ax.plot(diag["rmse_profile"][:, target_idx], height, label="RMSE", color="#7570b3", linewidth=2.0)
            ax.plot(diag["mae_profile"][:, target_idx], height, label="MAE", color="#e7298a", linewidth=2.0)
            ax.set_xlabel(target_name)
            if col_idx == 0:
                ax.set_ylabel("Height (m)")
            ax.set_title(f"{split_name.capitalize()} error profile", fontsize=10)
            ax.grid(True, alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "vertical_profile_errors_train_val_test.png", dpi=plot_dpi)
    plt.close(fig)

    if importance_results.get("enabled", False):
        feature_names = importance_results["feature_names"]
        fig, axes = plt.subplots(
            len(target_names),
            1,
            figsize=(6.0, 2.4 * len(target_names)),
        )
        if len(target_names) == 1:
            axes = [axes]

        for ax, target_name in zip(axes, target_names):
            ranking = importance_results["by_target"][target_name]
            delta_rmse = np.asarray(ranking["delta_rmse_mean"], dtype=np.float32)
            order = np.argsort(delta_rmse)[::-1]
            ordered_labels = [feature_names[idx] for idx in order]
            ordered_values = delta_rmse[order]
            ordered_uncertainty = np.asarray(ranking["delta_rmse_std"], dtype=np.float32)[order]

            ax.barh(ordered_labels, ordered_values, xerr=ordered_uncertainty, color="#4c78a8", alpha=0.9)
            ax.invert_yaxis()
            ax.set_xlabel("Validation RMSE increase after permutation")
            ax.set_title(f"Permutation importance for {target_name}", fontsize=10)
            ax.grid(True, axis="x", alpha=0.25)

        fig.tight_layout()
        fig.savefig(output_dir / "feature_permutation_importance.png", dpi=plot_dpi)
        plt.close(fig)


def build_profile_diagnostics(
    data: PreparedData,
    predictions: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    sample_mode = data.metadata["sample_mode"]
    height_levels = np.asarray(data.metadata["height_levels_m"], dtype=np.float32)
    profile_diagnostics: dict[str, dict[str, np.ndarray]] = {}

    for split_name in ("train", "val", "test"):
        split = getattr(data, split_name)
        prediction = predictions[split_name]
        truth = split.targets_raw

        if sample_mode == "column":
            diff = prediction - truth
            truth_mean = truth.mean(axis=0)
            prediction_mean = prediction.mean(axis=0)
            bias_mean = diff.mean(axis=0)
            rmse_profile = np.sqrt(np.mean(diff**2, axis=0))
            mae_profile = np.mean(np.abs(diff), axis=0)
            sample_count = np.full(height_levels.shape, truth.shape[0], dtype=np.int32)
        elif sample_mode == "pointwise":
            z_index = split.coords["z_index"].astype(np.int32)
            num_levels = len(height_levels)
            num_targets = truth.shape[-1]
            truth_mean = np.full((num_levels, num_targets), np.nan, dtype=np.float32)
            prediction_mean = np.full((num_levels, num_targets), np.nan, dtype=np.float32)
            bias_mean = np.full((num_levels, num_targets), np.nan, dtype=np.float32)
            rmse_profile = np.full((num_levels, num_targets), np.nan, dtype=np.float32)
            mae_profile = np.full((num_levels, num_targets), np.nan, dtype=np.float32)
            sample_count = np.zeros(num_levels, dtype=np.int32)

            for level_idx in range(num_levels):
                mask = z_index == level_idx
                if not np.any(mask):
                    continue
                level_truth = truth[mask]
                level_prediction = prediction[mask]
                level_diff = level_prediction - level_truth
                sample_count[level_idx] = int(mask.sum())
                truth_mean[level_idx] = level_truth.mean(axis=0)
                prediction_mean[level_idx] = level_prediction.mean(axis=0)
                bias_mean[level_idx] = level_diff.mean(axis=0)
                rmse_profile[level_idx] = np.sqrt(np.mean(level_diff**2, axis=0))
                mae_profile[level_idx] = np.mean(np.abs(level_diff), axis=0)
        else:
            raise ValueError(f"Unsupported sample_mode {sample_mode!r}.")

        profile_diagnostics[split_name] = {
            "height_m": height_levels,
            "truth_mean": truth_mean.astype(np.float32),
            "prediction_mean": prediction_mean.astype(np.float32),
            "bias_mean": bias_mean.astype(np.float32),
            "rmse_profile": rmse_profile.astype(np.float32),
            "mae_profile": mae_profile.astype(np.float32),
            "sample_count": sample_count.astype(np.int32),
        }

    return profile_diagnostics


def write_profile_diagnostics_netcdf(
    path: Path,
    profile_diagnostics: dict[str, dict[str, np.ndarray]],
    data: PreparedData,
) -> None:
    split_names = list(profile_diagnostics.keys())
    height = profile_diagnostics[split_names[0]]["height_m"]
    target_names = np.asarray(data.target_names, dtype="<U64")

    dataset = xr.Dataset(
        data_vars={
            "truth_mean": (
                ("split", "z", "target"),
                np.stack([profile_diagnostics[name]["truth_mean"] for name in split_names], axis=0),
            ),
            "prediction_mean": (
                ("split", "z", "target"),
                np.stack([profile_diagnostics[name]["prediction_mean"] for name in split_names], axis=0),
            ),
            "bias_mean": (
                ("split", "z", "target"),
                np.stack([profile_diagnostics[name]["bias_mean"] for name in split_names], axis=0),
            ),
            "rmse_profile": (
                ("split", "z", "target"),
                np.stack([profile_diagnostics[name]["rmse_profile"] for name in split_names], axis=0),
            ),
            "mae_profile": (
                ("split", "z", "target"),
                np.stack([profile_diagnostics[name]["mae_profile"] for name in split_names], axis=0),
            ),
            "sample_count": (
                ("split", "z"),
                np.stack([profile_diagnostics[name]["sample_count"] for name in split_names], axis=0),
            ),
        },
        coords={
            "split": np.asarray(split_names, dtype="<U16"),
            "z": np.arange(len(height), dtype=np.int32),
            "height_m": ("z", height.astype(np.float32)),
            "target": target_names,
        },
        attrs={
            "description": "Mean vertical-profile diagnostics for emulator predictions versus LES targets.",
            "sample_mode": data.metadata["sample_mode"],
        },
    )
    dataset.to_netcdf(path, engine="netcdf4")


def compute_permutation_importance(
    model: nn.Module,
    data: PreparedData,
    scalers: dict[str, Any],
    metrics: dict[str, dict[str, dict[str, float]]],
    device: torch.device,
    batch_size: int,
    importance_config: dict[str, Any],
) -> dict[str, Any]:
    enabled = bool(importance_config.get("enabled", True))
    if not enabled:
        return {"enabled": False}

    split_name = importance_config.get("split", "val")
    repeats = int(importance_config.get("repeats", 3))
    max_samples = importance_config.get("max_samples")
    max_samples = None if max_samples is None else int(max_samples)
    seed = int(importance_config.get("seed", 17))
    rng = np.random.default_rng(seed)

    split = getattr(data, split_name)
    features = split.features
    targets = split.targets_raw

    if max_samples is not None and len(features) > max_samples:
        selection = np.sort(rng.choice(len(features), size=max_samples, replace=False))
        features = features[selection]
        targets = targets[selection]

    baseline_prediction = predict_split(model, features, scalers, device, batch_size)
    baseline_metrics = {
        target_name: regression_metrics(targets[..., idx], baseline_prediction[..., idx])
        for idx, target_name in enumerate(data.target_names)
    }

    by_target: dict[str, dict[str, list[float]]] = {
        target_name: {
            "baseline_rmse": baseline_metrics[target_name]["rmse"],
            "baseline_mae": baseline_metrics[target_name]["mae"],
            "baseline_r2": baseline_metrics[target_name]["r2"],
            "delta_rmse_mean": [],
            "delta_rmse_std": [],
            "delta_mae_mean": [],
            "delta_mae_std": [],
            "delta_r2_mean": [],
            "delta_r2_std": [],
        }
        for target_name in data.target_names
    }

    for feature_idx, feature_name in enumerate(data.feature_names):
        deltas_by_target = {
            target_name: {"rmse": [], "mae": [], "r2": []}
            for target_name in data.target_names
        }

        for _ in range(repeats):
            permuted_features = permute_feature(features, feature_idx, rng)
            permuted_prediction = predict_split(model, permuted_features, scalers, device, batch_size)

            for target_idx, target_name in enumerate(data.target_names):
                permuted_metrics = regression_metrics(targets[..., target_idx], permuted_prediction[..., target_idx])
                deltas_by_target[target_name]["rmse"].append(
                    permuted_metrics["rmse"] - baseline_metrics[target_name]["rmse"]
                )
                deltas_by_target[target_name]["mae"].append(
                    permuted_metrics["mae"] - baseline_metrics[target_name]["mae"]
                )
                deltas_by_target[target_name]["r2"].append(
                    permuted_metrics["r2"] - baseline_metrics[target_name]["r2"]
                )

        for target_name in data.target_names:
            by_target[target_name]["delta_rmse_mean"].append(float(np.mean(deltas_by_target[target_name]["rmse"])))
            by_target[target_name]["delta_rmse_std"].append(float(np.std(deltas_by_target[target_name]["rmse"])))
            by_target[target_name]["delta_mae_mean"].append(float(np.mean(deltas_by_target[target_name]["mae"])))
            by_target[target_name]["delta_mae_std"].append(float(np.std(deltas_by_target[target_name]["mae"])))
            by_target[target_name]["delta_r2_mean"].append(float(np.mean(deltas_by_target[target_name]["r2"])))
            by_target[target_name]["delta_r2_std"].append(float(np.std(deltas_by_target[target_name]["r2"])))

    return {
        "enabled": True,
        "method": "permutation_importance",
        "split": split_name,
        "repeats": repeats,
        "max_samples": max_samples,
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "by_target": by_target,
        "note": "Positive delta_rmse and delta_mae indicate that permuting the predictor degraded skill. More negative delta_r2 indicates greater importance.",
    }


def permute_feature(features: np.ndarray, feature_idx: int, rng: np.random.Generator) -> np.ndarray:
    permuted = np.array(features, copy=True)
    order = rng.permutation(len(features))
    feature_values = np.array(permuted[..., feature_idx], copy=True)
    permuted[..., feature_idx] = feature_values[order]
    return permuted


if __name__ == "__main__":
    main()
