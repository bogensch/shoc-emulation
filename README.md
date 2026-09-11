# CASS Training Emulator

This repository now contains a PyTorch training pipeline for the
coarse-grained CASS emulator problem described in
`GENESIS_CASS_ML_CONTEXT.md`.

Current defaults match the constraints you gave:

- targets are only `w_prime_2` and `w_prime_3`
- `ustar` is not used as an input
- TKE is not used as an input
- resolved coarse-grid `w_wind` is also excluded by default so the first
  baseline relies on the mean thermodynamic and horizontal-wind state

## Files

- `train_emulator.py`: main training entry point
- `configs/baseline_mlp.yaml`: baseline experiment config
- `configs/column_context.yaml`: vertical-column-context experiment config
- `emulator_training/data.py`: NetCDF loading, feature construction,
  normalization, metrics
- `emulator_training/model.py`: pointwise MLP and column-context model definitions

## Baseline feature set

Direct 3-D predictors:

- `theta_liq`
- `qt`
- `qc`
- `rho`
- `pressure`
- `u_wind`
- `v_wind`
- `c_frac`

Broadcast 2-D predictors:

- `shf`
- `lhf`

Derived predictors:

- `height_m`
- `height_norm`
- `dtheta_liq_dz`
- `dqt_dz`
- `du_wind_dz`
- `dv_wind_dz`

## Multi-day dataset discovery

The default configs now discover all LES training days under:

`/p/lustre5/bogensch/ERF_shcu_ensemble`

Each usable day is expected at:

`shcu_50m_<yyyymmdd>/post_processed_output/`

with the two NetCDF files:

- `training_inputs_coarse_grained.3.2km.nc`
- `training_targets_coarse_grained_3.2km.nc`

The loader scans the ensemble root, keeps only case directories that contain
both files, and trains on the combined multi-day dataset.

To curate the day set without changing code, edit the config:

- leave `data.days` unset to auto-discover every usable day
- set `data.days` to an explicit list of `yyyymmdd` values if you want a fixed subset
- add entries to `data.exclude_days` to drop specific days from the discovered set

Legacy single-file configs using `data.input_path` and `data.target_path` are
still supported, but the shipped configs now use ensemble discovery by default.

## Day-level split

Train/validation/test splits are now assigned by entire LES day, not by time
windows within the same day. The default fractions are:

- 75% of days: train
- 12.5% of days: validation
- 12.5% of days: independent test

With the current 80-day ensemble, that yields:

- 60 training days
- 10 validation days
- 10 test days

By default, days are shuffled before splitting so the partition is reproducible
but not tied to directory sort order. The split is controlled by:

- `split.shuffle_days`
- `split.seed`

The resolved day lists and sample counts for each split are written to
`metadata.json` in every training output directory.

## Training-speed defaults

The shipped configs now use `training.train_shuffle_mode: once` to randomize
training order once up front instead of building a fresh full-dataset shuffle
every epoch.

The default batch sizes are also tuned upward to reduce optimizer steps on the
80-day dataset:

- `configs/baseline_mlp.yaml`: `training.batch_size: 65536`
- `configs/column_context.yaml`: `training.batch_size: 1024`

Supported shuffle modes are:

- `per_epoch`: reshuffle the training split every epoch
- `once`: shuffle the training split once before the first epoch
- `none`: preserve the original sample order

## Example usage

```bash
python3 train_emulator.py --config configs/baseline_mlp.yaml
```

For the explicit vertical-column-context model:

```bash
python3 train_emulator.py --config configs/column_context.yaml
```

For a quick smoke test:

```bash
python3 train_emulator.py \
  --config configs/column_context.yaml \
  --epochs 2 \
  --max-samples-per-split 200
```

Artifacts are written under `outputs/` and include:

- trained checkpoint
- resolved config
- normalization metadata
- scalar metrics
- split metadata including the resolved train/validation/test day lists
- training history plot
- mean vertical profile plots for train/validation/test
- vertical-profile diagnostics in NetCDF form (`vertical_profile_diagnostics.nc`)
- validation-set permutation importance in JSON form (`feature_importance.json`)
- permutation-importance ranking plot (`feature_permutation_importance.png`)

## Notes

- `w_prime_2` is trained with a `log1p` transform before standardization so
  the inverse-transformed predictions remain nonnegative.
- `w_prime_3` is trained with ordinary standardization.
- Feature importance is computed with validation-set permutation importance:
  one predictor is shuffled at a time and the change in validation skill is
  recorded for `w_prime_2` and `w_prime_3`.
- `configs/baseline_mlp.yaml` trains independent `(time, z, y, x)` samples.
- `configs/column_context.yaml` trains on entire coarse columns with shape
  `(z, feature)` and uses a 1-D convolutional network along height so each
  level can use neighboring vertical context.
