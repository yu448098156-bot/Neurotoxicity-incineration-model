# Author release checklist

- [ ] Confirm that the source-preserving CatBoost profile (500 iterations, depth 6, seed 42) is the intended final model, rather than merely the historical scatter-plot profile.
- [ ] Compare `reference_results/model_comparison.csv` with the manuscript table; do not relabel new metrics as historical exact reproduction.
- [ ] Update SHAP/ICE/Y-randomization figures to the outputs of the SAME final model, or document separate historical models rather than conflating them.
- [ ] Resolve the manuscript applicability-domain method separately; residual coverage is not AD.
- [ ] Supply the original feature-selection/tuning records if those procedures remain in the manuscript. This package uses the supplied fixed twenty descriptors and fixed parameters.
- [ ] Confirm toxicity endpoint units and route of administration against source experimental records. CSVs lack route-level provenance; no such records are inferred here.
- [ ] Add the real Zenodo dataset DOI to README and `CITATION.cff.example`/publication metadata as appropriate.
- [ ] Confirm all code/data rights and add licence files; then finalise the citation metadata.
- [ ] Run `python -m unittest discover -s tests -v` and `python -m neurotox verify` in a fresh environment.
- [ ] Review the contents for unwanted files, commit the repository, and tag the code version used for the final manuscript.

None of these items authorises modifying raw values to match a plotted annotation.
