#!/usr/bin/env python3
"""Hybrid physics-ML (PIML) model: GPR predicts the three constitutive parameters
(sigma_peak, strain_peak, E0) from the mix design (nS, PVA); the Carreira-Chu
uniaxial law then generates the full curve. Blind mixtures only - no leakage."""
import os
import numpy as np, pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pipeline as P

OUT = P.OUT
NAME = 'Hybrid PIML (GPR + Carreira-Chu)'
df, pr, curves = P.build_dataset()
inblind = lambda r: (r.nS, r.PVA) in P.TEST_COMBOS
blind = df.apply(inblind, axis=1)
B_te, B_tr = df[blind], df[~blind]
p_te, p_tr = pr[pr.apply(inblind, axis=1)], pr[~pr.apply(inblind, axis=1)]

fits = {}
for tgt in ['sigma_peak', 'strain_peak_um_m', 'E0_GPa']:
    gp = make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=Ck(1.0) * RBF([1., 1.]) + WhiteKernel(1e-3), normalize_y=True, random_state=42))
    gp.fit(p_tr[['nS', 'PVA']].values, p_tr[tgt].values)
    fits[tgt] = gp
    print(f'  {tgt:18s} measured={p_te[tgt].round(2).values}  GPR={gp.predict(p_te[["nS","PVA"]].values).round(2)}')

pos = {ix: i for i, ix in enumerate(B_te.index)}
yh = np.zeros(len(B_te))
for k, g in B_te.groupby(['nS', 'PVA']):
    X = np.array([[k[0], k[1]]])
    sp = float(fits['sigma_peak'].predict(X)[0])
    ep = float(fits['strain_peak_um_m'].predict(X)[0]) / 1e4        # um/m -> % strain
    E0 = float(fits['E0_GPa'].predict(X)[0]) * 1000.0              # GPa -> MPa
    e, s = P.carreira_chu(sp, ep / 100.0, E0)                      # law expects strain fraction
    yh[[pos[i] for i in g.index]] = np.interp(g.strain_pct.values / 100.0, e, s)

m = P.metrics(B_te.stress_MPa.values, yh)
m.update(model=NAME, split='B_blind_mixtures', n_train=len(B_tr), n_test=len(B_te))
res = pd.read_csv(os.path.join(OUT, 'benchmark_metrics.csv'))
res = res[~res.model.str.contains('Hybrid PIML')]
pd.concat([res, pd.DataFrame([m])]).to_csv(os.path.join(OUT, 'benchmark_metrics.csv'), index=False)

diag = P.physical_diagnostics(B_te, yh, P.fit_modulus_surface(pr, set(zip(B_tr.nS, B_tr.PVA))))
diag['model'] = NAME
dd = pd.read_csv(os.path.join(OUT, 'physical_diagnostics.csv'))
dd = dd[~dd.model.str.contains('Hybrid PIML')]
pd.concat([dd, pd.DataFrame([diag])]).to_csv(os.path.join(OUT, 'physical_diagnostics.csv'), index=False)

z = {k: np.load(os.path.join(OUT, 'predictions.npz'), allow_pickle=True)[k]
     for k in np.load(os.path.join(OUT, 'predictions.npz'), allow_pickle=True).files}
z['B_blind_mixtures||' + NAME] = yh
np.savez(os.path.join(OUT, 'predictions.npz'), **z)
print('HYBRID blind metrics:', {k: round(float(v), 4) for k, v in m.items() if isinstance(v, float)},
      {k: round(float(v), 4) for k, v in diag.items()})
