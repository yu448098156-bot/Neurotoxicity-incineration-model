"""Fast unit/integration tests. Full CV/LOO/permutations are not run in CI."""
import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from neurotox.io import ROOT,load_config,read_csv,sha256
from neurotox.data import load_inputs,compute_one
from neurotox.models import build_model
from neurotox.metrics import regression_metrics
from neurotox.persistence import load_native,native_predict
from neurotox.predict import predict

class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_config();cls.train,cls.test=load_inputs(cls.cfg)

    def test_fixed_split_counts(self):
        self.assertEqual((len(self.train.y),len(self.test.y)),(346,87))

    def test_input_bytes_unchanged(self):
        for name,digest in self.cfg['expected_input_sha256'].items():
            self.assertEqual(sha256(ROOT/'data'/name),digest)

    def test_no_canonical_structure_overlap(self):
        self.assertFalse(set(self.train.canonical)&set(self.test.canonical))

    def test_finite_twenty_descriptors(self):
        self.assertEqual(self.train.X.shape,(346,20))
        self.assertTrue(np.isfinite(self.train.X).all())
        self.assertTrue(np.isfinite(self.test.X).all())

    def test_descriptor_values_match_legacy_function(self):
        source=ast.parse((ROOT/'legacy/RF.py').read_text(encoding='utf-8-sig'))
        function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='calculate_descriptors')
        namespace={'Descriptors':Descriptors}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'RF_descriptor_definition','exec'),namespace)
        for row,x in zip(self.train.records[:10],self.train.X[:10]):
            old=namespace['calculate_descriptors'](Chem.MolFromSmiles(row['SMILES']))
            np.testing.assert_array_equal(x,[old[f] for f in self.cfg['features']])

    def test_source_estimator_hyperparameters(self):
        # Every original explicit scientific parameter is retained. Thread controls are operational.
        for name in ['RF','GBR','ETR','SVR','KNN','XGBoost']:
            filename=self.cfg['models'][name]['source']
            tree=ast.parse((ROOT/'legacy'/filename).read_text(encoding='utf-8-sig'))
            assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='base_model' for t in n.targets))
            old={k.arg:ast.literal_eval(k.value) for k in assignment.value.keywords if k.arg not in ['n_jobs']}
            self.assertEqual(old,self.cfg['models'][name]['params'])

    def test_metrics(self):
        m=regression_metrics([1,2,3],[1,2,3])
        self.assertEqual(m,{'R2':1.,'RMSE':0.,'MAE':0.})

    def test_invalid_smiles_rejected(self):
        with self.assertRaises(ValueError):compute_one('',self.cfg['features'])

    def test_fold_scaler_uses_only_training_partition(self):
        # Pure unit-test toy array; never used as study data.
        X=np.array([[0.],[1.],[2.],[1000.]]);y=np.array([0.,1.,2.,3.])
        model=build_model('SVR',self.cfg)
        model.fit(X[:3],y[:3])
        self.assertEqual(float(model.named_steps['scaler'].mean_[0]),1.)

    def test_all_models_clone(self):
        for name in self.cfg['model_order']:
            p=build_model(name,self.cfg)
            self.assertEqual(clone(p).named_steps['scaler'].get_params(),StandardScaler().get_params())

    def test_catboost_adapter_matches_native(self):
        from catboost import CatBoostRegressor
        from neurotox.catboost_adapter import CatBoostAdapter
        params={'iterations':3,'depth':2,'random_seed':42,'verbose':False,'allow_writing_files':False}
        x=self.train.X[:20];y=self.train.y[:20]
        wrapper=CatBoostAdapter(params,1).fit(x,y)
        native=CatBoostRegressor(**params,thread_count=1).fit(x,y)
        np.testing.assert_array_equal(wrapper.predict(x),native.predict(x))

    def test_saved_model_reproduces_reference_predictions(self):
        bundle=ROOT/'models/final_catboost'
        _,rows=read_csv(ROOT/'reference_results/CatBoost_predictions_test.csv')
        np.testing.assert_allclose(native_predict(bundle,self.test.X),[float(r['predicted_pLD50']) for r in rows],rtol=1e-10,atol=1e-10)

    def test_predict_preserves_invalid_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with (p/'input.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['Name','SMILES'])
                w.writerow(['valid',self.test.records[0]['SMILES']]);w.writerow(['missing',''])
            predict(p/'input.csv',ROOT/'models/final_catboost',p/'out.csv')
            _,rows=read_csv(p/'out.csv')
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['prediction_status'],'predicted')
            self.assertEqual(rows[1]['prediction_status'],'invalid_input')
            self.assertEqual(rows[1]['predicted_pLD50'],'')

if __name__=='__main__':unittest.main()
