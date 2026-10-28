#!/usr/bin/env python3
"""Train the representative ML / DL / PINN surrogates on the blind-mixture TRAINING
set and predict the full grid of all 24 compositions - used by the
"predicted versus experimental" figure and for inference-latency measurement."""
import os, json, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import pipeline as P

OUT = P.OUT
df, pr, curves = P.build_dataset()
blind = df.apply(lambda r: (r.nS, r.PVA) in P.TEST_COMBOS, axis=1)
B_tr = df[~blind]

Xs = B_tr[['nS', 'PVA', 'strain_pct']].values.astype(float)
ys = B_tr['stress_MPa'].values.astype(float)
Xall = df[['nS', 'PVA', 'strain_pct']].values.astype(float)

coef = P.fit_modulus_surface(pr, set(zip(B_tr.nS, B_tr.PVA)))
res, timing = {}, {}

# ------------------------------------------------------------------ ML
t0 = time.time()
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.9,
                   colsample_bytree=0.9, random_state=42, verbosity=0).fit(Xs, ys)
res['XGBoost'] = xgb.predict(Xall); timing['XGBoost'] = time.time() - t0

# ------------------------------------------------------------------ DL
t0 = time.time()
mlp = make_pipeline(StandardScaler(), MLPRegressor(
    hidden_layer_sizes=(64, 64, 32), activation='tanh', solver='adam',
    max_iter=6000, learning_rate_init=2e-3, random_state=42)).fit(Xs, ys)
res['Deep ANN'] = mlp.predict(Xall); timing['Deep ANN'] = time.time() - t0

# ------------------------------------------------------------------ PINN (3-seed ensemble, selected weights)
t0 = time.time()
acc = np.zeros(len(df))
for s in (0, 1, 2):
    m = P.train_pinn(B_tr, pr, coef, seed=s, lam_E=0.0, lam_dmg=0.0, lam_sm=0.0)
    acc += m.predict(Xall, m.scX, m.scY)
res['PINN'] = acc / 3.0; timing['PINN'] = time.time() - t0

# ------------------------------------------------------------------ inference latency: one full curve (60 pts)
lat = {}
for name, arr in res.items():
    n_curves, n_pts = 24, 60
    Xc = np.column_stack([np.full(n_pts, 2.0), np.full(n_pts, 1.0),
                          np.linspace(0, 0.3, n_pts)])
    t0 = time.time()
    for _ in range(200):
        if name == 'XGBoost':
            xgb.predict(Xc)
        elif name == 'Deep ANN':
            mlp.predict(Xc)
        else:
            m.predict(Xc, m.scX, m.scY)
    lat[name] = (time.time() - t0) / 200 * 1e3          # ms per 60-point curve

np.savez(os.path.join(OUT, 'pred_all.npz'),
         **{k: v for k, v in res.items()}, dfall_ns=df.nS.values,
         dfall_pva=df.PVA.values, dfall_strain_pct=df.strain_pct.values,
         dfall_stress=df.stress_MPa.values)
json.dump({'train_seconds': timing, 'inference_ms_per_curve': lat},
          open(os.path.join(OUT, 'pred_all_meta.json'), 'w'), indent=2)
print('train seconds:', {k: round(v, 1) for k, v in timing.items()})
print('inference ms / 60-point curve:', {k: round(v, 3) for k, v in lat.items()})
for k, v in res.items():
    Bte = df[blind]
    print(f'  {k:10s} blind R2 = {1 - np.sum((Bte.stress_MPa.values - v[Bte.index.values]) ** 2) / np.sum((Bte.stress_MPa.values - Bte.stress_MPa.mean()) ** 2):.4f}')
