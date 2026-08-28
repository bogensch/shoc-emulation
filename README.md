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

## Development split

You currently have a single CASS day. Because a day-level split is not yet
possible, the code uses contiguous time blocks from that day:

- first 70% of times: train
- next 15%: validation
- final 15%: test

This is only a development/debug split. Once multiple days are available, the
split logic should be changed to split by entire days rather than by time
windows within one day.

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
- training history plot
- mean vertical profile plots for train/validation
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
- Once multiple days are available, the split logic should be updated to split
  by day rather than by time windows within a single day.
