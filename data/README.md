# Model input data

`training_set.csv` and `test_set.csv` are byte-for-byte copies of the author-supplied files (346 and 87 rows). The split, row order, SMILES and pLD50 values are frozen in `config/models.json` by SHA-256.

- `No.`: original author identifier; not a reindexed rank.
- `CAS`: supplied registry identifier, retained as text; not independently reassigned or verified against SMILES.
- `SMILES`: supplied structure, used to compute the fixed 20 RDKit descriptors. No salt removal, tautomer standardization or molecule replacement is performed.
- `pLD50`: supplied modelling response, used at the stored precision. The manuscript defines this on a negative-log molar scale, but these CSVs do not carry the original dose conversion or exposure-route provenance. The package does not infer these missing experimental annotations.

`prediction_example.csv` contains three real test-set structures and no outcome column. It is a demonstration input, not new validation data.

Other study data and mass spectrometry files belong in the separate Zenodo record. Data licensing, the full toxicity-record provenance and the actual Zenodo DOI must be confirmed by the authors.
