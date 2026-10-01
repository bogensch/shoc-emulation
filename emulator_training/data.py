from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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

LOG1P_NONNEGATIVE_TOLERANCE = 1.0e-6


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


@dataclass(frozen=True)
class ResolvedCase:
    day: str
    case_name: str
    input_path: Path
    target_path: Path


def log_data(message: str) -> None:
    print(f"[DATA] {message}", flush=True)


def load_datasets(input_path: str | Path, target_path: str | Path) -> tuple[xr.Dataset, xr.Dataset]:
    inputs = xr.open_dataset(input_path)
    targets = xr.open_dataset(target_path)

    expected_dims = ("time", "z", "y", "x")
    for dim in expected_dims:
        if inputs.sizes[dim] != targets.sizes[dim]:
            raise ValueError(f"Dimension mismatch for {dim}: {inputs.sizes[dim]} vs {targets.sizes[dim]}")

    return inputs, targets


def prepare_dataset(config: dict[str, Any], max_samples_per_split: int | None = None) -> PreparedData:
    if "ensemble_root" in config["data"]:
        return prepare_multi_day_dataset(config, max_samples_per_split=max_samples_per_split)
    return prepare_single_file_dataset(config, max_samples_per_split=max_samples_per_split)


def prepare_single_file_dataset(config: dict[str, Any], max_samples_per_split: int | None = None) -> PreparedData:
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


def prepare_multi_day_dataset(config: dict[str, Any], max_samples_per_split: int | None = None) -> PreparedData:
    data_cfg = config["data"]
    split_cfg = config["split"]
    sample_mode = data_cfg.get("sample_mode", "pointwise")
    split_seed = int(split_cfg.get("seed", config["training"]["seed"]))
    rng = np.random.default_rng(split_seed)

    cases = discover_case_datasets(data_cfg)
    log_data(f"Discovered {len(cases)} usable day(s) under {data_cfg['ensemble_root']}")
    split_days = build_day_splits(
        days=[case.day for case in cases],
        train_fraction=float(split_cfg["train_fraction"]),
        val_fraction=float(split_cfg["val_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
        seed=split_seed,
        shuffle_days=bool(split_cfg.get("shuffle_days", True)),
    )
    log_data(
        "Day split counts: "
        f"train={len(split_days['train'])}, val={len(split_days['val'])}, test={len(split_days['test'])}"
    )
    day_to_split = {day: split_name for split_name, days in split_days.items() for day in days}

    split_members: dict[str, list[SplitArrays]] = {"train": [], "val": [], "test": []}
    split_case_names: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    times_per_day: dict[str, int] = {}
    feature_names: list[str] | None = None
    target_names = list(data_cfg["targets"])
    height_levels_m: np.ndarray | None = None
    reference_sizes: dict[str, int] | None = None
    reference_height: np.ndarray | None = None

    for day_index, case in enumerate(cases):
        split_name = day_to_split[case.day]
        log_data(f"Loading day {day_index + 1}/{len(cases)}: {case.case_name} -> {split_name}")
        inputs, targets = load_datasets(case.input_path, case.target_path)
        try:
            current_sizes = {dim: int(inputs.sizes[dim]) for dim in ("z", "y", "x")}
            current_height = inputs["height"].values.astype(np.float32, copy=False)

            if reference_sizes is None:
                reference_sizes = current_sizes
                reference_height = np.array(current_height, copy=True)
                height_levels_m = current_height.astype(np.float32, copy=False)
            else:
                if current_sizes != reference_sizes:
                    raise ValueError(
                        f"Case {case.case_name} has grid sizes {current_sizes}, expected {reference_sizes}."
                    )
                if not np.allclose(current_height, reference_height):
                    raise ValueError(f"Case {case.case_name} has height levels that differ from the reference grid.")

            feature_arrays, current_feature_names = build_feature_arrays(
                inputs=inputs,
                predictor_3d=data_cfg["predictor_3d"],
                predictor_2d=data_cfg["predictor_2d"],
                derived_predictors=data_cfg["derived_predictors"],
            )
            if feature_names is None:
                feature_names = current_feature_names
            elif current_feature_names != feature_names:
                raise ValueError(f"Case {case.case_name} resolved a different feature ordering.")

            target_arrays = [targets[name].values.astype(np.float32, copy=False) for name in target_names]
            raw_features = np.stack(feature_arrays, axis=-1)
            raw_targets = np.stack(target_arrays, axis=-1)

            if sample_mode == "pointwise":
                prepared = flatten_pointwise_case(
                    inputs=inputs,
                    raw_features=raw_features,
                    raw_targets=raw_targets,
                    day_index=day_index,
                    day_name=case.day,
                )
            elif sample_mode == "column":
                prepared = flatten_column_case(
                    inputs=inputs,
                    raw_features=raw_features,
                    raw_targets=raw_targets,
                    day_index=day_index,
                    day_name=case.day,
                )
            else:
                raise ValueError(f"Unsupported sample_mode {sample_mode!r}.")

            split_members[split_name].append(prepared)
            split_case_names[split_name].append(case.case_name)
            times_per_day[case.day] = int(inputs.sizes["time"])
            log_data(
                f"Prepared {case.case_name}: time={inputs.sizes['time']}, "
                f"samples={len(prepared.features)}, split={split_name}"
            )
        finally:
            inputs.close()
            targets.close()

    log_data("Concatenating split arrays")
    train = concatenate_split_arrays(split_members["train"], max_samples_per_split, rng)
    val = concatenate_split_arrays(split_members["val"], max_samples_per_split, rng)
    test = concatenate_split_arrays(split_members["test"], max_samples_per_split, rng)
    log_data(
        "Finished dataset assembly: "
        f"train_samples={len(train.features)}, val_samples={len(val.features)}, test_samples={len(test.features)}"
    )

    metadata = {
        "dataset_mode": "multi_day",
        "sample_mode": sample_mode,
        "num_days": len(cases),
        "num_levels": int(reference_sizes["z"]) if reference_sizes is not None else None,
        "num_y": int(reference_sizes["y"]) if reference_sizes is not None else None,
        "num_x": int(reference_sizes["x"]) if reference_sizes is not None else None,
        "height_levels_m": height_levels_m.astype(float).tolist() if height_levels_m is not None else [],
        "feature_names": feature_names,
        "target_names": target_names,
        "ensemble_root": str(data_cfg["ensemble_root"]),
        "case_prefix": str(data_cfg.get("case_prefix", "shcu_50m_")),
        "input_filename": str(data_cfg.get("input_filename", "training_inputs_coarse_grained.3.2km.nc")),
        "target_filename": str(data_cfg.get("target_filename", "training_targets_coarse_grained_3.2km.nc")),
        "days": [case.day for case in cases],
        "case_names": [case.case_name for case in cases],
        "case_directories": [str(case.input_path.parent.parent) for case in cases],
        "times_per_day": times_per_day,
        "split_method": "day",
        "split_seed": split_seed,
        "shuffle_days": bool(split_cfg.get("shuffle_days", True)),
        "split_days": split_days,
        "split_case_names": split_case_names,
        "split_day_counts": {name: len(days) for name, days in split_days.items()},
        "split_sample_counts": {
            "train": int(len(train.features)),
            "val": int(len(val.features)),
            "test": int(len(test.features)),
        },
    }

    return PreparedData(
        train=train,
        val=val,
        test=test,
        feature_names=feature_names or [],
        target_names=target_names,
        metadata=metadata,
    )


def discover_case_datasets(data_cfg: dict[str, Any]) -> list[ResolvedCase]:
    root = Path(data_cfg["ensemble_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Ensemble root was not found: {root}")

    case_prefix = str(data_cfg.get("case_prefix", "shcu_50m_"))
    post_processed_subdir = str(data_cfg.get("post_processed_subdir", "post_processed_output"))
    input_filename = str(data_cfg.get("input_filename", "training_inputs_coarse_grained.3.2km.nc"))
    target_filename = str(data_cfg.get("target_filename", "training_targets_coarse_grained_3.2km.nc"))
    excluded_days = {str(day) for day in data_cfg.get("exclude_days", [])}
    explicit_days = data_cfg.get("days")
    pattern = re.compile(rf"^{re.escape(case_prefix)}(\d{{8}})$")

    cases: list[ResolvedCase] = []
    if explicit_days is not None:
        seen_days: set[str] = set()
        for raw_day in explicit_days:
            day = str(raw_day)
            if day in seen_days:
                raise ValueError(f"Duplicate day {day!r} in data.days.")
            seen_days.add(day)
            if day in excluded_days:
                continue
            case_name = f"{case_prefix}{day}"
            input_path = root / case_name / post_processed_subdir / input_filename
            target_path = root / case_name / post_processed_subdir / target_filename
            if not input_path.is_file() or not target_path.is_file():
                raise FileNotFoundError(
                    f"Missing training NetCDFs for requested day {day}: {input_path} and/or {target_path}"
                )
            cases.append(
                ResolvedCase(
                    day=day,
                    case_name=case_name,
                    input_path=input_path,
                    target_path=target_path,
                )
            )
        if not cases:
            raise ValueError("No usable days remained after applying data.days and data.exclude_days.")
        return cases

    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        day = match.group(1)
        if day in excluded_days:
            continue
        input_path = path / post_processed_subdir / input_filename
        target_path = path / post_processed_subdir / target_filename
        if not input_path.is_file() or not target_path.is_file():
            continue
        cases.append(
            ResolvedCase(
                day=day,
                case_name=path.name,
                input_path=input_path,
                target_path=target_path,
            )
        )

    if not cases:
        raise ValueError(f"No training cases with both NetCDF files were found under {root}.")
    return cases


def build_day_splits(
    days: list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    shuffle_days: bool,
) -> dict[str, list[str]]:
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}")
    if len(days) < 3:
        raise ValueError("Day-level train/val/test splitting requires at least three days.")
    if len(set(days)) != len(days):
        raise ValueError("Duplicate day labels were provided for day splitting.")

    ordered_days = list(days)
    if shuffle_days:
        ordered_days = [ordered_days[idx] for idx in np.random.default_rng(seed).permutation(len(ordered_days))]

    split_names = ("train", "val", "test")
    counts = allocate_fractional_counts(
        len(ordered_days),
        fractions=np.asarray([train_fraction, val_fraction, test_fraction], dtype=np.float64),
        split_names=split_names,
    )

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": ordered_days[:train_end],
        "val": ordered_days[train_end:val_end],
        "test": ordered_days[val_end:],
    }


def allocate_fractional_counts(
    total_count: int,
    fractions: np.ndarray,
    split_names: tuple[str, ...],
) -> dict[str, int]:
    raw_counts = fractions * total_count
    counts = np.floor(raw_counts).astype(np.int32)
    remainder = int(total_count - counts.sum())
    priority = np.argsort(-(raw_counts - counts))
    for idx in priority[:remainder]:
        counts[idx] += 1

    if total_count >= len(split_names):
        for idx in np.where(counts == 0)[0]:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                raise ValueError("Unable to allocate at least one day to each split.")
            counts[donor] -= 1
            counts[idx] += 1

    return {name: int(counts[idx]) for idx, name in enumerate(split_names)}


def flatten_pointwise_case(
    inputs: xr.Dataset,
    raw_features: np.ndarray,
    raw_targets: np.ndarray,
    day_index: int,
    day_name: str,
) -> SplitArrays:
    time_hours = inputs["time_hours"].values.astype(np.float32)
    height = inputs["height"].values.astype(np.float32)
    time_index = np.arange(inputs.sizes["time"], dtype=np.int32)
    z_index = np.arange(inputs.sizes["z"], dtype=np.int32)
    y_index = np.arange(inputs.sizes["y"], dtype=np.int32)
    x_index = np.arange(inputs.sizes["x"], dtype=np.int32)

    mesh = np.meshgrid(time_index, z_index, y_index, x_index, indexing="ij")
    sample_count = mesh[0].size
    coords = {
        "day_index": np.full(sample_count, day_index, dtype=np.int32),
        "day_name": np.full(sample_count, day_name, dtype=f"<U{max(len(day_name), 1)}"),
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
    return SplitArrays(features=flat_features, targets_raw=flat_targets, coords=coords)


def flatten_column_case(
    inputs: xr.Dataset,
    raw_features: np.ndarray,
    raw_targets: np.ndarray,
    day_index: int,
    day_name: str,
) -> SplitArrays:
    features = np.transpose(raw_features, (0, 2, 3, 1, 4))
    targets = np.transpose(raw_targets, (0, 2, 3, 1, 4))

    time_index = np.arange(inputs.sizes["time"], dtype=np.int32)
    y_index = np.arange(inputs.sizes["y"], dtype=np.int32)
    x_index = np.arange(inputs.sizes["x"], dtype=np.int32)
    mesh = np.meshgrid(time_index, y_index, x_index, indexing="ij")
    sample_count = mesh[0].size
    coords = {
        "day_index": np.full(sample_count, day_index, dtype=np.int32),
        "day_name": np.full(sample_count, day_name, dtype=f"<U{max(len(day_name), 1)}"),
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
    return SplitArrays(features=flat_features, targets_raw=flat_targets, coords=coords)


def concatenate_split_arrays(
    members: list[SplitArrays],
    max_samples: int | None,
    rng: np.random.Generator,
) -> SplitArrays:
    if not members:
        raise ValueError("Encountered an empty split after dataset preparation.")

    features = np.concatenate([member.features for member in members], axis=0)
    targets = np.concatenate([member.targets_raw for member in members], axis=0)
    coords = {
        name: np.concatenate([member.coords[name] for member in members], axis=0)
        for name in members[0].coords
    }

    if max_samples is not None and len(features) > max_samples:
        selection = np.sort(rng.choice(len(features), size=max_samples, replace=False))
        features = features[selection]
        targets = targets[selection]
        coords = {name: values[selection] for name, values in coords.items()}

    return SplitArrays(features=features, targets_raw=targets, coords=coords)


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
        values = values.astype(np.float32, copy=False)
        min_value = float(np.min(values))
        if min_value < -LOG1P_NONNEGATIVE_TOLERANCE:
            raise ValueError(
                "log1p_standardize requires nonnegative target values. "
                f"Found minimum value {min_value:.6e}, which is below the clipping tolerance "
                f"{LOG1P_NONNEGATIVE_TOLERANCE:.1e}."
            )
        return np.log1p(np.clip(values, a_min=0.0, a_max=None)).astype(np.float32)
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
