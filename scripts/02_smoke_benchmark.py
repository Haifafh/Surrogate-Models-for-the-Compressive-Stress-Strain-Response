#!/usr/bin/env python3
"""Fast, self-contained check: refit three baseline surrogates on the blind split.

This is a smoke test, not the full benchmark. It reproduces the qualitative
ordering of the paper (gradient-boosted trees strongest, linear regression
weakest) in a few seconds. Absolute values differ slightly from the manuscript
because the full benchmark uses tuned settings and multi-seed ensembling.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import data as D
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = D.load_dataset()
tr, te = D.split_B(df)
FEAT = ['nS', 'PVA', 'strain_um_m']
Xtr, ytr = tr[FEAT].to_numpy(float), tr['stress_MPa'].to_numpy(float)
Xte, yte = te[FEAT].to_numpy(float), te['stress_MPa'].to_numpy(float)
print(f'Split B: {len(tr)} train points, {len(te)} blind test points from {te.groupby(["nS","PVA"]).ngroups} held-out mixtures')

models = {
    'Linear regression': LinearRegression(),
    'Random forest': RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
}
try:
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                                     random_state=42, n_jobs=-1)
except ImportError:
    print('xgboost not installed - skipping')

print(f'\n{"model":24s} {"R2":>8s} {"RMSE (MPa)":>11s}')
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    rmse = float(np.sqrt(np.mean((p - yte) ** 2)))
    print(f'{name:24s} {r2_score(yte, p):8.4f} {rmse:11.2f}')
print('\nNote: the manuscript reports XGBoost at R2=0.9895 / RMSE=2.22 MPa on Split B.')
