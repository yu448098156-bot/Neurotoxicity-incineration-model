# Neurotoxicity regression for waste-incineration candidate screening

**Version 1.0.0 — unified, source-based reference implementation**

Python code, the original training/test CSVs, and a saved CatBoost prediction model supporting neurotoxicity-oriented chemical screening. The repository compares seven fixed-descriptor regression algorithms and provides reproducible evaluation, model persistence, new-SMILES prediction, SHAP interpretation and ICE analysis.

**Scope:** model code and the model's required inputs only. Raw mass-spectrometry files, environmental concentrations, NEQ/ToxPi inputs, and global emissions are distributed separately through the authors' Zenodo record. **Authors: insert the actual Zenodo DOI before release.**

This is a refactor of author-supplied scripts, not a claim that every earlier manuscript metric or diagram is reproduced exactly. Method changes and unresolved provenance questions are documented in [Methods and changes](docs/METHODS_AND_CHANGES.md). The saved model was fitted on **346 training compounds only**; the **87 test compounds are not included in its training**.

## Quick start

Use a fresh Python environment. The supplied model/reference run was tested on **CPython 3.13.5, Linux**, with the exact versions in `requirements.txt`. CPU execution is sufficient; no GPU is required. Windows/macOS use is intended but was not tested in this release.

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

From the **repository root**, install the locked dependencies and verify the supplied model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m neurotox verify
```

The tests include descriptor/source-parameter checks, immutable input checks, pipeline preprocessing, API compatibility, invalid-row handling and saved-model prediction reproduction. They are not the full expensive validation analysis. If an environment cannot resolve the locked dependencies, use the tested Python version rather than silently substituting a different learner.

## Predict without retraining

```bash
python -m neurotox predict --input data/prediction_example.csv --out outputs/example_predictions.csv
```

Replace the input path with a CSV containing a `SMILES` column. Other ID columns are preserved. A `pLD50` column is **not needed** for inference. The command loads `models/final_catboost/model.cbm` and the matching `preprocessing.json`; it does not fit another scaler/model or need experimental toxicity values. Invalid input rows remain in the output with an error message.

The output is predicted **pLD50**, not an experimentally established neurotoxicity classification. Descriptor-range flags are simple training-range checks, **not a validated applicability domain**. Do not apply an additional scaler to the input SMILES/descriptors.

## Reproduce the full workflow

```bash
python run_pipeline.py --out outputs/reproduction --jobs 2 --loo --permutations 100
```

This trains all seven regressors, performs common five-fold training CV and actual leave-one-out validation, saves each fitted pipeline, exports the final CatBoost, computes TreeSHAP and the four selected ICE profiles, runs 100 training-label permutations and saves each figure separately. The output directory must be new or empty; prior results are not deleted. Full LOO and permutations are slow. Omit `--loo` for a faster run; omitted LOO fields are explicitly labelled `not_run`.

The fixed hyperparameters are from the supplied scripts. **This does not rerun Bayesian optimisation or feature selection.** Those original procedures/search records were not provided.

### Run individual stages

```bash
# Data/overlap audit only, without training
python -m neurotox audit --out outputs/audit

# Fit all seven models, shared five-fold CV, and actual LOO; save outputs
python -m neurotox compare --out outputs/comparison --jobs 2 --loo

# Explain the model from THIS run, not a newly trained interpretation model
python -m neurotox explain --model-dir outputs/comparison/final_catboost --out outputs/explanation

# Reproducible training-only Y-randomization
python -m neurotox yrandom --model-dir outputs/comparison/final_catboost --out outputs/yrandom --iterations 100 --seed 42 --jobs 2

# Separate SVG/PNG panels from saved numeric CSVs
python -m neurotox plots --comparison outputs/comparison --explanation outputs/explanation --yrandom outputs/yrandom --out outputs/figures
```



## Seven models and source parameters

| Model                           | Source script            | Retained principal settings                                                |
| ------------------------------- | ------------------------ | -------------------------------------------------------------------------- |
| Random forest (RF)              | `RF.py`                  | 300 trees; unrestricted depth; seed 42                                     |
| Gradient boosting (GBR)         | `GBR.py`                 | 200 estimators; rate 0.05; depth 5; seed 42                                |
| Extra trees (ETR)               | `ETR.py`                 | 500 trees; unrestricted depth; leaf 1; split 2; no bootstrap; seed 42      |
| Support vector regression (SVR) | `SVR.py`                 | RBF; C=10; epsilon=0.1; gamma=scale                                        |
| K-nearest neighbours (KNN)      | `KNeighborsRegressor.py` | k=5; distance weighting; Minkowski metric                                  |
| XGBoost                         | `XGboost.py`             | 400 trees; rate 0.05; depth 4; row/column fractions 0.8; lambda=1; seed 42 |
| CatBoost                        | `catboost-figure1.py`    | 500 iterations; rate 0.05; depth 6; seed 42                                |

## Saved model files

`models/final_catboost/` contains:

- `model.cbm`: native fitted CatBoost model, recommended for the supplied prediction CLI.
- `preprocessing.json`: ordered descriptor names, training scaler and descriptive ranges.
- `metadata.json`: training/test hashes, parameters, versions and provenance.
- `pipeline.joblib`: trusted Python pipeline including scaler and CatBoost adapter.
- `bundle_checksums.json`: file-integrity checks.

Native prediction avoids loading pickle. Only load a `.joblib`/pickle artifact from a trusted source; such formats can execute code and require compatible library versions. All seven fitted pipelines are also saved by `compare` under the run's `models/<model>/` directory. The separately distributed full reference-run archive contains these seven pipelines. The GitHub package retains only the designated final model to keep the repository focused.

## Reference results and provenance

`reference_results/` contains small, model-related comparison/validation tables, exact test predictions, SHAP importance, ICE mean curves and numerical integrity checks from the refactored run. Large individual-curve exports are generated by the commands above and supplied in the separate run-output archive. No environmental dataset is duplicated here.

Important distinctions:

1. The historical prediction, SHAP and ICE scripts did not train the same CatBoost profile. This release uses one model throughout; interpretation plots and some metrics therefore differ from the historical figures.
2. The original residual threshold called “AD” is preserved only as `legacy_error_diagnostics_NOT_AD.csv`. It is not used as an applicability-domain filter or as a model-selection criterion.
3. The source `pLD50` values are used at their stored precision. These CSVs do not establish the original experimental route or the complete toxicity-record provenance.
4. There is no reconstructed historical hyperparameter-search or feature-selection pipeline. The model comparison/permutation tests are conditional on the supplied fixed features and settings.
5. Preserving the original train/test split does not turn this already-used test set into a newly independent external validation cohort.

See the [data dictionary](docs/DATA_DICTIONARY.md), [release checklist](docs/RELEASE_CHECKLIST.md) and [Chinese guide](README_zh-CN.md).
