#!/usr/bin/env python3
"""Extended evaluation metrics for the blind-mixture split.

Complements R2 / RMSE / MAE / MAPE with the statistics that are standard in
surrogate-model and hydrological-model validation but were missing from the first
draft: worst-case error (MaxAE), systematic bias (ME), normalised RMSE, Pearson r,
Willmott's index of agreement (d), Kling-Gupta efficiency (KGE), peak-strain error,
and inference latency. Also computes the prediction-interval coverage (PICP) of the
Gaussian-process surrogate.
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import pipeline as P

OUT = P.OUT
df, pr, curves = P.build_dataset()
blind = df.apply(lambda r: (r.nS, r.PVA) in P.TEST_COMBOS, axis=1)
B_te, B_tr = df[blind], df[~blind]
y = B_te.stress_MPa.values.astype(float)

z = np.load(os.path.join(OUT, 'predictions.npz'), allow_pickle=True)
preds = {tuple(k.split('||')): z[k] for k in z.files}
meta = json.load(open(os.path.join(OUT, 'pred_all_meta.json')))
lat = meta['inference_ms_per_curve']


def ext(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    resid = yhat - y
    ss_res = float(np.sum(resid ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    m = y > 1.0
    d = dict(
        R2=1 - ss_res / ss_tot,
        RMSE=rmse,
        MAE=float(np.mean(np.abs(resid))),
        MaxAE=float(np.max(np.abs(resid))),
        ME=float(np.mean(resid)),
        MAPE=float(np.mean(np.abs(resid[m] / y[m])) * 100),
        NRMSE=rmse / float(y.max() - y.min()),
        r=float(np.corrcoef(y, yhat)[0, 1]),
        rho=float(spearmanr(y, yhat).correlation),
        d=1 - ss_res / float(np.sum((np.abs(yhat - y.mean()) + np.abs(y - y.mean())) ** 2)),
    )
    alpha = np.std(yhat) / np.std(y); beta = np.mean(yhat) / np.mean(y)
    d['KGE'] = 1 - float(np.sqrt((d['r'] - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return d


def peak_errs(yhat):
    d = B_te.copy(); d['pred'] = yhat
    sp, se = [], []
    for k, g in d.groupby(['nS', 'PVA']):
        sp.append(abs(g.pred.max() - g.stress_MPa.max()) / g.stress_MPa.max() * 100)
        ip, jp = g.pred.idxmax(), g.stress_MPa.idxmax()
        se.append(abs(g.loc[ip, 'strain_um_m'] - g.loc[jp, 'strain_um_m']) / g.loc[jp, 'strain_um_m'] * 100)
    return float(np.mean(sp)), float(np.mean(se))


rows = []
for (sp_, nm), yhat in preds.items():
    if sp_ != 'B_blind_mixtures' or nm == '__idx__':
        continue
    r = ext(y, yhat)
    r['peak_stress_err'] , r['peak_strain_err'] = peak_errs(yhat)
    r['inference_ms'] = lat.get(nm, np.nan)
    r['model'] = nm
    rows.append(r)

# ---- Gaussian-process prediction-interval coverage (calibration of uncertainty)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(); Xtr = sc.fit_transform(B_tr[['nS', 'PVA', 'strain_pct']].values)
gp = GaussianProcessRegressor(kernel=Ck(1.0) * RBF([1., 1., 1.]) + WhiteKernel(1e-3),
                              normalize_y=True, random_state=42).fit(Xtr, B_tr.stress_MPa.values)
Xte = sc.transform(B_te[['nS', 'PVA', 'strain_pct']].values)
mu, sd = gp.predict(Xte, return_std=True)
picp95 = float(np.mean(np.abs(y - mu) <= 1.96 * sd) * 100)
mpiw = float(np.mean(2 * 1.96 * sd))
print(f'GP 95% interval: coverage PICP = {picp95:.1f}%  (nominal 95%),  mean width = {mpiw:.2f} MPa')
json.dump({'PICP95': picp95, 'MPIW95': mpiw}, open(os.path.join(OUT, 'gp_calibration.json'), 'w'), indent=2)

out = pd.DataFrame(rows)[['model', 'R2', 'RMSE', 'MAE', 'MaxAE', 'ME', 'MAPE', 'NRMSE',
                          'r', 'rho', 'd', 'KGE', 'peak_stress_err', 'peak_strain_err',
                          'inference_ms']]
out = out.sort_values('R2', ascending=False)
out.to_csv(os.path.join(OUT, 'metrics_extended.csv'), index=False)
pd.set_option('display.width', 250, 'display.max_columns', 40)
print(out.round(4).to_string(index=False))
