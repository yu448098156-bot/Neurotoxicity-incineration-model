# Output dictionary

| File | One row represents | Main columns / interpretation |
|---|---|---|
| `model_comparison.csv` | one algorithm | Training/test R², RMSE, MAE; shared five-fold validation means/s.d.; pooled CV Q²; actual LOO Q² when enabled. |
| `<model>_cv_folds.csv` | one fold | Fold ID, training/validation counts and metrics. |
| `cv_assignments.csv` | one training compound | Validation-fold assignment, fixed across models. |
| `<model>_predictions_test.csv` | one test compound | Source IDs, observed/predicted pLD50, signed residual `predicted-observed`, absolute error. |
| `<model>_predictions_training.csv` | one training compound | In-sample, out-of-fold and optionally LOO predictions; validation fold. |
| `legacy_error_diagnostics_NOT_AD.csv` | one model | Historical residual-threshold diagnostic, **not** applicability-domain coverage. |
| `save_reload_checks.csv` | one model | Maximum prediction difference before/after persistence. |
| `data_audit/split_manifest.csv` | one input record | Original ID, CSV row, SMILES, audit canonical SMILES and pLD50. |
| `data_audit/descriptors_raw.csv` | one input record | Twenty original-scale descriptor values in fixed order. |
| `data_audit/cross_split_descriptor_collisions.csv` | one pair | Exact training/test representation collisions; empty when none. |
| `SHAP_values.csv` | one compound/descriptor | Raw/scaled descriptor, signed SHAP, expected value and model prediction. |
| `SHAP_importance.csv` | one descriptor | Mean absolute SHAP and descriptive feature–SHAP association. |
| `ICE_<descriptor>.csv` | one grid coordinate | Scaled/raw x, PDP mean, one column for each training compound's curve. |
| `ICE_curve_ids.csv` | one training compound | Curve column name to source identifiers. |
| `ICE_PDP.csv` | one descriptor/grid point | Long-format coordinates and the average over all individual curves. |
| `ICE_curve_changes.csv` | one curve | Last-grid prediction minus first-grid prediction, not an estimated causal effect. |
| `Y_randomization.csv` | original or one permutation | In-sample R², mean five-fold R², pooled CV Q² and each fold R². |
| `permutation_indices.npz` | numeric matrix | Shape B × 346, zero-based label indices; read with `allow_pickle=False`. |
| `predictions.csv` | one new input row | Predicted pLD50 or error, plus descriptive training-range checks; no regulatory or clinical classification. |

CSV files use UTF-8 with BOM, one header row, no Excel formulas and no premature rounding. Empty LOO fields indicate “not run”, not zero. Model-native inputs are standardized with the saved training mean/scale; do not apply a second scaler.
