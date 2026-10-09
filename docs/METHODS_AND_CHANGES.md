# Scope, mathematical definitions and changes from the supplied code

## 1. What is source-preserving

The two CSVs remain byte-for-byte unchanged: 346 training records and 87 test records. The exact 20 RDKit descriptors, ordering, and explicit hyperparameters of the six newly supplied algorithms are retained in `config/models.json`. The CatBoost prediction profile is taken from `legacy/catboost-figure1.py`: 500 iterations, depth 6, learning rate 0.05 and seed 42. These settings are a traceable choice from the supplied files, not proof that they were the settings behind every historical figure or manuscript table.

No new labels, compounds, data split, feature selection or hyperparameter tuning is introduced. Source outcomes are used at their stored precision. In particular, the code does NOT reconstruct the Bayesian optimisation, SelectPercentile/RFE or training-label provenance described in the manuscript; those original records were not supplied. The present comparison is conditional on the provided fixed feature set and parameters.

## 2. Unified fitting and validation

Every model is a `StandardScaler -> regressor` pipeline. The scaler is fitted only to the training partition of each validation fold, never to test observations. The final saved model/scaler are fitted on the 346 training records only. The 87 test records are evaluated but are not used for fitting, parameter tuning, early stopping, or automated model selection.

All seven models use identical five-fold training partitions (`KFold`, shuffle=True, random_state=42); fold indices are exported. This common CV design is newly specified for the unified code. It is not claimed to reproduce previously unrecorded cross-validation partitions. LOO, when requested, fits a fresh scaler and model in every one of the 346 folds. An omitted LOO calculation is labelled `not_run`, never replaced with CV or in-sample scores.

Training and test R² = 1 - sum((y - prediction)^2)/sum((y - mean(y))^2). RMSE is the square root of the mean squared error; MAE is the mean absolute error. `Q2_test_equals_R2_test` is mathematically identical to test R² under this denominator, not an independent validation statistic. `Q2_CV_pooled` and `Q2_LOO` use all out-of-fold predictions and the overall training mean; the mean of five fold-level R² values is separately labelled. The fold-level standard deviation uses ddof=1. Adjusted training R² retains the source equation with p=20; it is labelled descriptive because 20 input descriptors do not represent the effective degrees of freedom of nonlinear ensembles.

## 3. One interpretation model, not three

The supplied historical scripts trained different CatBoost models: test scatter (500/depth6/seed42), SHAP (200/depth8/seed42) and ICE/Y-randomization (200/depth8/no explicitly recorded model seed). The unified analysis loads ONE saved prediction-profile CatBoost for predictions, TreeSHAP, ICE and the Y-randomization baseline. Thus new interpretation outputs can differ from previous diagrams. This is an explicit methodological harmonisation, not a silent attempt to reproduce historical numbers.

Native CatBoost TreeSHAP values are exported for all 346 training structures and all 20 descriptors. Additivity is checked numerically against the saved model predictions. Four ICE descriptors are fixed at the author's request: EState_VSA5, SlogP_VSA2, SMR_VSA4 and VSA_EState2. They are called selected descriptors, not automatically inferred top-four features. Each ICE grid uses all unique training values in standardized units. Other descriptors are held at each compound's observed values. Raw grid coordinates and the mean curve are exported with all individual curves. These are model-response analyses, not causal interventions or validated mechanisms.

## 4. Y-randomization

Training labels are permuted with a saved `numpy.random.default_rng(42)` state recipe. All zero-based permutation index arrays are also written, so the exact permutations are available. One full-training model and five fold-specific models are fitted per permutation, with the same CatBoost profile and training-only scaling. The test set is not used. The unpermuted baseline is checked against the saved native model. The empirical one-sided p-value for mean CV R² is (1 + number of permutation statistics >= original)/(B + 1).

This fixed-feature, fixed-parameter permutation assessment does not repeat the missing historical feature-selection/hyperparameter-search stages. It therefore assesses this fixed modelling specification, not the entire original discovery procedure. These are new seeded runs, not recovered historical randomized predictions.

## 5. Error coverage is NOT an applicability domain

The original six comparison scripts label a threshold of `1.5 * RMSE_test` as AD and report the fraction of observed residuals below that threshold. This requires the actual test responses and is a retrospective error statistic, not a chemical-space applicability-domain criterion for unseen compounds.

The calculation is retained ONLY in `legacy_error_diagnostics_NOT_AD.csv`; it does not affect fitting, model selection, target predictions or priority decisions. The unified code does not claim the manuscript's 93.1% AD result. Its original structural-distance/leverage implementation and thresholds were not supplied. `predict` reports the count of descriptors outside the training min–max range as a descriptive screen; this is explicitly NOT a validated AD or a safety/hazard classification.

## 6. Other transparent changes

- Removed automatic fallback from CatBoost to HistGradientBoostingRegressor. Missing dependencies now cause a visible error instead of silently changing the scientific model.
- Added a thin sklearn-compatible CatBoost wrapper because CatBoost 1.2.8 lacks the estimator-tags interface expected by sklearn 1.8. Fitting/prediction remain native CatBoost; a unit test verifies identical outputs.
- Refactored duplicated functions and gave every model its own output folder. The original scripts otherwise overwrite `metrics_summary.csv` and `predictions_test.csv`.
- Kept source outcome precision and all input rows; invalid training data now raise explicit errors rather than disappearing. Prediction inputs retain invalid rows with an error column.
- Added data checksums, structure/descriptor overlap audits, environment and model metadata, and post-save reload tests. Canonicalization is used for audit only, not as an undocumented structure replacement.
- Removed the original residual-standard-deviation band labelled “95% Prediction Band” from the new plotting entry point. That band did not implement a calibrated predictive interval. New scatter plots show data, a linear fit and the 1:1 reference.
- CPU thread limits and `allow_writing_files=False` are explicit operational settings. They do not introduce an alternate algorithm.

## 7. Publication boundaries

The repository is a reference implementation from supplied code, not an independent recreation of every historical figure, claimed optimum or published endpoint. Numerical differences should be reconciled with the manuscript rather than edited away. The response is pLD50 as supplied; predictions are screening outputs and not experimental confirmation of neurotoxicity, human disease risk or inhalation-route potency.

The authors should supply their code/data licence choices, real Zenodo DOI and final manuscript citation. They should also confirm the precise neurotoxicity endpoint/administration-route provenance, the model profile actually used for the 695 environmental predictions, and any feature-selection/tuning steps they continue to describe in Methods. Those environmental predictions are not recomputed here.
