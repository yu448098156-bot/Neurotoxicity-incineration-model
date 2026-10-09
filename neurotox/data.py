"""Compute the supplied fixed descriptor set without re-splitting or relabelling."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from .io import ROOT, read_csv, sha256, write_csv, write_json

@dataclass
class Dataset:
    records: list[dict]
    X: np.ndarray
    y: np.ndarray | None
    canonical: list[str]
    features: list[str]

    @property
    def ids(self) -> list[str]:
        return [str(r['record_id']) for r in self.records]

def compute_one(smiles: str, features: list[str]) -> tuple[np.ndarray, str]:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('SMILES is missing or blank')
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError('SMILES could not be parsed by RDKit')
    try:
        values = np.array([getattr(Descriptors, f)(mol) for f in features], dtype=float)
    except Exception as exc:
        raise ValueError(f'Descriptor calculation failed: {exc}') from exc
    if not np.isfinite(values).all():
        names = [f for f, v in zip(features, values) if not np.isfinite(v)]
        raise ValueError(f'Non-finite descriptor values: {names}')
    return values, Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def load_dataset(path: Path, features: list[str], split: str, require_target: bool = True) -> Dataset:
    headers, rows = read_csv(path)
    needed = {'SMILES', 'pLD50'} if require_target else {'SMILES'}
    if not needed.issubset(headers):
        raise ValueError(f'{path.name}: required columns {sorted(needed)}; found {headers}')
    if not rows:
        raise ValueError(f'{path.name}: no data rows')
    X, y, canonical, records, errors = [], [], [], [], []
    for i, row in enumerate(rows):
        try:
            values, key = compute_one(row['SMILES'], features)
            target = float(row['pLD50']) if require_target else None
            if require_target and not np.isfinite(target):
                raise ValueError('pLD50 is not finite')
        except (ValueError, TypeError) as exc:
            errors.append(f'CSV row {i+2}: {exc}')
            continue
        records.append({'record_id':f'{split}_{i+1:04d}', 'split':split,
                        'source_csv_row':i+2, 'source_No':row.get('No.', ''),
                        'CAS':row.get('CAS',''), 'SMILES':row['SMILES']})
        X.append(values); y.append(target); canonical.append(key)
    if errors:
        raise ValueError(f'{path.name}: invalid records; nothing was silently dropped.\n'+'\n'.join(errors))
    return Dataset(records, np.asarray(X), np.asarray(y, dtype=float) if require_target else None,
                   canonical, features)

def load_inputs(cfg: dict) -> tuple[Dataset, Dataset]:
    train_path, test_path = ROOT/cfg['train_file'], ROOT/cfg['test_file']
    for path in [train_path, test_path]:
        expected = cfg.get('expected_input_sha256',{}).get(path.name)
        if expected and sha256(path) != expected:
            raise ValueError(f'Input checksum mismatch: {path}. Use a separate config for changed data.')
    train = load_dataset(train_path,cfg['features'],'training')
    test = load_dataset(test_path,cfg['features'],'test')
    for split, ds in [('training',train),('test',test)]:
        expected = cfg.get('expected_counts',{}).get(split)
        if expected is not None and len(ds.records)!=expected:
            raise ValueError(f'{split}: expected {expected} records, found {len(ds.records)}')
    return train,test

def data_audit(train: Dataset, test: Dataset, output: Path, cfg: dict) -> dict:
    allrecords = train.records+test.records
    allkeys = train.canonical+test.canonical
    allX = np.vstack([train.X,test.X])
    buckets = defaultdict(list)
    for row,key in zip(allrecords,allkeys):
        buckets[key].append(row)
    duplicate_rows = []
    for key,rows in buckets.items():
        if len(rows)>1:
            duplicate_rows.extend({'canonical_smiles':key,'record_id':r['record_id'],
                                   'split':r['split'],'cross_split':len({v['split'] for v in rows})>1}
                                  for r in rows)
    # Exact equality of the FIXED model representation is informative even when structures differ.
    collisions = []
    for i in range(len(train.X)):
        matches=np.flatnonzero(np.all(test.X == train.X[i],axis=1))
        for j in matches:
            collisions.append({'training_record_id':train.ids[i],'test_record_id':test.ids[j],
                               'same_canonical_structure':train.canonical[i]==test.canonical[j],
                               'training_pLD50':float(train.y[i]),'test_pLD50':float(test.y[j])})
    report={'train_n':len(train.X),'test_n':len(test.X),'descriptor_n':len(train.features),
            'finite_descriptors':bool(np.isfinite(allX).all()),
            'train_unique_canonical_smiles':len(set(train.canonical)),
            'test_unique_canonical_smiles':len(set(test.canonical)),
            'cross_split_canonical_overlap':len(set(train.canonical)&set(test.canonical)),
            'cross_split_exact_descriptor_pairs':len(collisions),
            'constant_training_descriptors':[f for f,v in zip(train.features,np.ptp(train.X,axis=0)) if v==0],
            'input_sha256':{p:sha256(ROOT/cfg[k]) for p,k in [('training_set.csv','train_file'),('test_set.csv','test_file')]},
            'target_handling':'Source pLD50 and source split retained; no inferred route, missing values, deduplication or rescaling.',
            'assessment_scope':'Canonical SMILES and exact descriptor audit only; not a chemical-identity or toxicological-provenance validation.'}
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'input_audit.json',report)
    write_csv(output/'canonical_duplicates.csv',['canonical_smiles','record_id','split','cross_split'],duplicate_rows)
    write_csv(output/'cross_split_descriptor_collisions.csv',['training_record_id','test_record_id','same_canonical_structure','training_pLD50','test_pLD50'],collisions)
    headers=['record_id','split','source_csv_row','source_No','CAS','SMILES','canonical_smiles','pLD50']
    write_csv(output/'split_manifest.csv',headers,[{**r,'canonical_smiles':key,'pLD50':float(target)}
              for r,key,target in zip(allrecords,allkeys,np.r_[train.y,test.y])])
    write_csv(output/'descriptors_raw.csv',['record_id','split']+train.features,
              [{'record_id':r['record_id'],'split':r['split'],**dict(zip(train.features,x))} for r,x in zip(allrecords,allX)])
    return report
