from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


DERIVED_FEATURES = {
    "height_m",
    "height_norm",
    "dtheta_liq_dz",
    "dqt_dz",
    "du_wind_dz",
    "dv_wind_dz",
}


@dataclass
class SplitArrays:
    features: np.ndarray
    targets_raw: np.ndarray
    coords: dict[str, np.ndarray]


@dataclass
class PreparedData:
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    feature_names: list[str]
    target_names: list[str]
    metadata: dict[str, Any]


def load_datasets(input_path: str | Path, target_path: str | Path) -> tuple[xr.Dataset, xr.Dataset]:
    inputs = xr.open_dataset(input_path)
    targets = xr.open_dataset(target_path)

    expected_dims = ("time", "z", "y", "x")
    for dim in expected_dims:
        if inputs.sizes[dim] != targets.sizes[dim]:
            raise ValueError(f"Dimension mismatch for {dim}: {inputs.sizes[dim]} vs {targets.sizes[dim]}")

    return inputs, targets


def prepare_dataset(config: dict[str, Any], max_samples_per_split: int | None = None) -> PreparedData:
    data_cfg = config["data"]
    split_cfg = config["split"]
    seed = int(config["training"]["seed"])
    sample_mode = data_cfg.get("sample_mode", "pointwise")

    inputs, targets = load_datasets(data_cfg["input_path"], data_cfg["target_path"])

    feature_arrays, feature_names = build_feature_arrays(
        inputs=inputs,
        predictor_3d=data_cfg["predictor_3d"],
        predictor_2d=data_cfg["predictor_2d"],
        derived_predictors=data_cfg["derived_predictors"],
    )
    target_arrays = [targets[name].values.astype(np.float32, copy=False) for name in data_cfg["targets"]]
    target_names = list(data_cfg["targets"])

    raw_features = np.stack(feature_arrays, axis=-1)
    raw_targets = np.stack(target_arrays, axis=-1)

    split_indices = build_time_splits(
        num_times=inputs.sizes["time"],
        train_fraction=float(split_cfg["train_fraction"]),
        val_fraction=float(split_cfg["val_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
    )

    rng = np.random.default_rng(seed)
    if sample_mode == "pointwise":
        train, val, test = prepare_pointwise_splits(
            inputs=inputs,
            raw_features=raw_features,
            raw_targets=raw_targets,
            split_indices=split_indices,
            max_samples_per_split=max_samples_per_split,
            rng=rng,
        )
    elif sample_mode == "column":
        train, val, test = prepare_column_splits(
            inputs=inputs,
            raw_features=raw_features,
            raw_targets=raw_targets,
            split_indices=split_indices,
            max_samples_per_split=max_samples_per_split,
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported sample_mode {sample_mode!r}.")

    metadata = {
        "sample_mode": sample_mode,
        "num_times": int(inputs.sizes["time"]),
        "num_levels": int(inputs.sizes["z"]),
        "num_y": int(inputs.sizes["y"]),
        "num_x": int(inputs.sizes["x"]),
        "height_levels_m": inputs["height"].values.astype(float).tolist(),
        "split_indices": {name: values.tolist() for name, values in split_indices.items()},
        "feature_names": feature_names,
        "target_names": target_names,
        "input_path": str(data_cfg["input_path"]),
        "target_path": str(data_cfg["target_path"]),
    }

    inputs.close()
    targets.close()

    return PreparedData(
        train=train,
        val=val,
        test=test,
        feature_names=feature_names,
        target_names=target_names,
        metadata=metadata,
    )


def prepare_pointwise_splits(
    inputs: xr.Dataset,
    raw_features: np.ndarray,
    raw_targets: np.ndarray,
    split_indices: dict[str, np.ndarray],
    max_samples_per_split: int | None,
    rng: np.random.Generator,
) -> tuple[SplitArrays, SplitArrays, SplitArrays]:
    time_hours = inputs["time_hours"].values.astype(np.float32)
    height = inputs["height"].values.astype(np.float32)
    time_index = np.arange(inputs.sizes["time"], dtype=np.int32)
    z_index = np.arange(inputs.sizes["z"], dtype=np.int32)
    y_index = np.arange(inputs.sizes["y"], dtype=np.int32)
    x_index = np.arange(inputs.sizes["x"], dtype=np.int32)

    mesh = np.meshgrid(time_index, z_index, y_index, x_index, indexing="ij")
    coords = {
        "time_index": mesh[0].reshape(-1),
        "z_index": mesh[1].reshape(-1),
        "y_index": mesh[2].reshape(-1),
        "x_index": mesh[3].reshape(-1),
        "time_hours": time_hours[mesh[0]].reshape(-1),
        "height_m": height[mesh[1]].reshape(-1),
    }

    flat_features = raw_features.reshape(-1, raw_features.shape[-1])
    flat_targets = raw_targets.reshape(-1, raw_targets.shape[-1])

    finite_mask = np.isfinite(flat_features).all(axis=1) & np.isfinite(flat_targets).all(axis=1)
    flat_features = flat_features[finite_mask]
    flat_targets = flat_targets[finite_mask]
    coords = {name: values[finite_mask] for name, values in coords.items()}

    sample_time = coords["time_index"]
    train_mask = np.isin(sample_time, split_indices["train"])
    val_mask = np.isin(sample_time, split_indices["val"])
    test_mask = np.isin(sample_time, split_indices["test"])

    train = subset_split(flat_features, flat_targets, coords, train_mask, max_samples_per_split, rng)
    val = subset_split(flat_features, flat_targets, coords, val_mask, max_samples_per_split, rng)
    test = subset_split(flat_features, flat_targets, coords, test_mask, max_samples_per_split, rng)
    return train, val, test


def prepare_column_splits(
    inputs: xr.Dataset,
    raw_features: np.ndarray,
    raw_targets: np.ndarray,
    split_indices: dict[str, np.ndarray],
    max_samples_per_split: int | None,
    rng: np.random.Generator,
) -> tuple[SplitArrays, SplitArrays, SplitArrays]:
    features = np.transpose(raw_features, (0, 2, 3, 1, 4))
    targets = np.transpose(raw_targets, (0, 2, 3, 1, 4))

    time_index = np.arange(inputs.sizes["time"], dtype=np.int32)
    y_index = np.arange(inputs.sizes["y"], dtype=np.int32)
    x_index = np.arange(inputs.sizes["x"], dtype=np.int32)
    mesh = np.meshgrid(time_index, y_index, x_index, indexing="ij")
    coords = {
        "time_index": mesh[0].reshape(-1),
        "y_index": mesh[1].reshape(-1),
        "x_index": mesh[2].reshape(-1),
        "time_hours": inputs["time_hours"].values.astype(np.float32)[mesh[0]].reshape(-1),
    }

    flat_features = features.reshape(-1, features.shape[3], features.shape[4])
    flat_targets = targets.reshape(-1, targets.shape[3], targets.shape[4])

    finite_mask = np.isfinite(flat_features).all(axis=(1, 2)) & np.isfinite(flat_targets).all(axis=(1, 2))
    flat_features = flat_features[finite_mask]
    flat_targets = flat_targets[finite_mask]
    coords = {name: values[finite_mask] for name, values in coords.items()}

    sample_time = coords["time_index"]
    train_mask = np.isin(sample_time, split_indices["train"])
    val_mask = np.isin(sample_time, split_indices["val"])
    test_mask = np.isin(sample_time, split_indices["test"])

    train = subset_split(flat_features, flat_targets, coords, train_mask, max_samples_per_split, rng)
    val = subset_split(flat_features, flat_targets, coords, val_mask, max_samples_per_split, rng)
    test = subset_split(flat_features, flat_targets, coords, test_mask, max_samples_per_split, rng)
    return train, val, test


def build_feature_arrays(
    inputs: xr.Dataset,
    predictor_3d: list[str],
    predictor_2d: list[str],
    derived_predictors: list[str],
) -> tuple[list[np.ndarray], list[str]]:
    feature_arrays: list[np.ndarray] = []
    feature_names: list[str] = []

    for name in predictor_3d:
        if name not in inputs:
            raise KeyError(f"Requested 3D predictor {name!r} was not found in the inputs dataset.")
        feature_arrays.append(inputs[name].values.astype(np.float32, copy=False))
        feature_names.append(name)

    for name in predictor_2d:
        if name not in inputs:
            raise KeyError(f"Requested 2D predictor {name!r} was not found in the inputs dataset.")
        arr = inputs[name].values.astype(np.float32, copy=False)
        feature_arrays.append(np.broadcast_to(arr[:, None, :, :], inputs["theta_liq"].shape))
        feature_names.append(name)

    for name in derived_predictors:
        if name not in DERIVED_FEATURES:
            raise KeyError(f"Unsupported derived predictor {name!r}.")
        feature_arrays.append(build_derived_feature(inputs, name))
        feature_names.append(name)

    return feature_arrays, feature_names


def build_derived_feature(inputs: xr.Dataset, name: str) -> np.ndarray:
    shape = inputs["theta_liq"].shape
    height = inputs["height"].values.astype(np.float32)
    height_4d = np.broadcast_to(height[None, :, None, None], shape)

    if name == "height_m":
        return height_4d
    if name == "height_norm":
        denom = max(float(height.max()), 1.0)
        return height_4d / denom
    if name == "dtheta_liq_dz":
        return vertical_gradient(inputs["theta_liq"].values, height)
    if name == "dqt_dz":
        return vertical_gradient(inputs["qt"].values, height)
    if name == "du_wind_dz":
        return vertical_gradient(inputs["u_wind"].values, height)
    if name == "dv_wind_dz":
        return vertical_gradient(inputs["v_wind"].values, height)
    raise KeyError(f"Unsupported derived predictor {name!r}.")


def vertical_gradient(values: np.ndarray, height: np.ndarray) -> np.ndarray:
    return np.gradient(values.astype(np.float32), height.astype(np.float32), axis=1).astype(np.float32)


def build_time_splits(
    num_times: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, np.ndarray]:
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}")

    train_end = max(1, int(round(num_times * train_fraction)))
    val_end = max(train_end + 1, int(round(num_times * (train_fraction + val_fraction))))
    val_end = min(val_end, num_times)

    indices = np.arange(num_times, dtype=np.int32)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("The configured split leaves an empty validation or test segment.")

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def subset_split(
    features: np.ndarray,
    targets: np.ndarray,
    coords: dict[str, np.ndarray],
    mask: np.ndarray,
    max_samples: int | None,
    rng: np.random.Generator,
) -> SplitArrays:
    split_features = features[mask]
    split_targets = targets[mask]
    split_coords = {name: values[mask] for name, values in coords.items()}

    if max_samples is not None and len(split_features) > max_samples:
        selection = np.sort(rng.choice(len(split_features), size=max_samples, replace=False))
        split_features = split_features[selection]
        split_targets = split_targets[selection]
        split_coords = {name: values[selection] for name, values in split_coords.items()}

    return SplitArrays(features=split_features, targets_raw=split_targets, coords=split_coords)


def fit_standard_scaler(values: np.ndarray) -> dict[str, np.ndarray]:
    flat_values = values.reshape(-1, values.shape[-1])
    mean = flat_values.mean(axis=0)
    std = flat_values.std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_standard_scaler(values: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return ((values - scaler["mean"]) / scaler["std"]).astype(np.float32)


def invert_standard_scaler(values: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return (values * scaler["std"] + scaler["mean"]).astype(np.float32)


def fit_target_transforms(
    raw_targets: np.ndarray,
    target_names: list[str],
    transform_config: dict[str, str],
) -> list[dict[str, Any]]:
    transforms: list[dict[str, Any]] = []
    for idx, name in enumerate(target_names):
        raw = raw_targets[..., idx]
        transform_name = transform_config.get(name, "standardize")
        transformed = forward_target_transform(raw, transform_name)
        scaler = fit_standard_scaler(transformed[..., None])
        transforms.append(
            {
                "name": name,
                "transform": transform_name,
                "mean": float(scaler["mean"][0]),
                "std": float(scaler["std"][0]),
            }
        )
    return transforms


def transform_targets(raw_targets: np.ndarray, transforms: list[dict[str, Any]]) -> np.ndarray:
    transformed_columns = []
    for idx, transform in enumerate(transforms):
        column = forward_target_transform(raw_targets[..., idx], transform["transform"])
        normalized = (column - transform["mean"]) / transform["std"]
        transformed_columns.append(normalized.astype(np.float32))
    return np.stack(transformed_columns, axis=-1)


def inverse_transform_targets(normalized_targets: np.ndarray, transforms: list[dict[str, Any]]) -> np.ndarray:
    restored_columns = []
    for idx, transform in enumerate(transforms):
        physical_space = normalized_targets[..., idx] * transform["std"] + transform["mean"]
        restored = inverse_target_transform(physical_space, transform["transform"])
        restored_columns.append(restored.astype(np.float32))
    return np.stack(restored_columns, axis=-1)


def forward_target_transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "standardize":
        return values.astype(np.float32)
    if transform_name == "log1p_standardize":
        if np.any(values < 0.0):
            raise ValueError("log1p_standardize requires nonnegative target values.")
        return np.log1p(values).astype(np.float32)
    raise KeyError(f"Unsupported target transform {transform_name!r}.")


def inverse_target_transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "standardize":
        return values.astype(np.float32)
    if transform_name == "log1p_standardize":
        return np.expm1(values).astype(np.float32)
    raise KeyError(f"Unsupported target transform {transform_name!r}.")


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = truth.astype(np.float64).reshape(-1)
    prediction = prediction.astype(np.float64).reshape(-1)
    diff = prediction - truth
    mse = np.mean(diff**2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    centered_truth = truth - truth.mean()
    centered_prediction = prediction - prediction.mean()
    denom = np.sqrt(np.sum(centered_truth**2) * np.sum(centered_prediction**2))
    corr = float(np.sum(centered_truth * centered_prediction) / denom) if denom > 0.0 else float("nan")
    sst = np.sum((truth - truth.mean()) ** 2)
    r2 = float(1.0 - np.sum(diff**2) / sst) if sst > 0.0 else float("nan")
    return {"rmse": rmse, "mae": mae, "bias": bias, "corr": corr, "r2": r2}
