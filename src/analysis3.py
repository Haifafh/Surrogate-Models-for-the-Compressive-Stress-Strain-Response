#!/usr/bin/env python3
"""Reviewer-response analysis (part 2).
P2: Table-5 peak sensitivity; P3: PDP of peak stress/strain vs nS and PVA,
GP conformal recalibration, hyperparameter sensitivity."""
import os, sys, json, warnings, copy
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT = os.path.expanduser('~/workspace/paper')
sys.path.insert(0, ROOT); os.chdir(ROOT)
import pipeline as P
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
OUT='out'

df, pr, curves = P.build_dataset()
blind = df.apply(lambda r: (r.nS, r.PVA) in P.TEST_COMBOS, axis=1)
B_tr, B_te = df[~blind], df[blind]
coefB = P.fit_modulus_surface(pr, set(zip(B_tr.nS, B_tr.PVA)))
Xtr, ytr = B_tr[['nS','PVA','strain_pct']].values, B_tr.stress_MPa.values
Xte, yte = B_te[['nS','PVA','strain_pct']].values, B_te.stress_MPa.values
combos = sorted(set(zip(df.nS, df.PVA)))

# published companion-paper Table 5 peak values
T5 = {(0,0.5):(60.0,2300),(0,1.0):(57.0,2400),(0,1.5):(53.0,2450),(0,2.0):(51.0,2500),
      (1,0.5):(61.0,2450),(1,1.0):(58.0,2600),(1,1.5):(55.0,2650),(1,2.0):(52.0,2700),
      (2,0.5):(63.0,2600),(2,1.0):(60.0,2750),(2,1.5):(57.0,2800),(2,2.0):(55.0,2850),
      (3,0.5):(65.0,2750),(3,1.0):(62.0,2850),(3,1.5):(60.0,2900),(3,2.0):(58.0,2950),
      (4,0.5):(66.0,2850),(4,1.0):(63.0,2950),(4,1.5):(61.0,3000),(4,2.0):(59.0,3050),
      (5,0.5):(65.0,2750),(5,1.0):(62.0,2850),(5,1.5):(60.0,2900),(5,2.0):(57.0,2950)}

# fine prediction grid for every mixture
grid = []
for ns,pva in combos:
    for s in np.linspace(0.0, 3.2, 80):
        grid.append((ns,pva,s))
G = np.array(grid, float)
print('grid', G.shape)

# ---------------------------------------------------------------- fit models
Z = P.model_zoo()
fitted = {}
for nm in ['XGBoost','Gaussian Process','Random Forest','Deep ANN (MLP 64-64-32)','Linear Regression']:
    m = copy.deepcopy(Z[nm]); m.fit(Xtr, ytr); fitted[nm] = m
calib = json.load(open(os.path.join(OUT,'config_manifest.json')))
kw = calib['arm_deployed']; opt = {k:kw[k] for k in ('lr','batch','epochs')}
ctl = dict(opt, lam_E=0.0, lam_dmg=0.0, lam_sm=0.0)
acc = np.zeros(len(G))
for s in (0,1,2):
    m = P.train_pinn(B_tr, pr, coefB, seed=s, **kw)
    acc += m.predict(G, m.scX, m.scY)
fitted['PINN (physics-informed)'] = None
pinnG = acc/3.0
print('lambda deployed:', [kw['lam_E'],kw['lam_dmg'],kw['lam_sm']])

# ---------------------------------------------------------------- peak sensitivity
rows = []
for nm, m in fitted.items():
    pred = pinnG if m is None else m.predict(G)
    g = pd.DataFrame({'nS':G[:,0],'PVA':G[:,1],'s':G[:,2],'p':pred})
    pk = g.loc[g.groupby(['nS','PVA']).p.idxmax()].set_index(['nS','PVA']).p
    for (ns,pva), v in pk.items():
        d5, e5 = T5[(ns,pva)]
        dg = float(pr[(pr.nS==ns)&(pr.PVA==pva)].sigma_peak.iloc[0])
        rows.append(dict(model=nm, nS=ns, PVA=pva, pred_peak=float(v), table5_peak=d5,
                         digitised_peak=dg))
pk_df = pd.DataFrame(rows)
pk_df['err_vs_table5'] = pk_df.pred_peak - pk_df.table5_peak
pk_df['err_vs_digitised'] = pk_df.pred_peak - pk_df.digitised_peak
pk_df.to_csv(os.path.join(OUT,'table5_sensitivity.csv'), index=False)

def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))
sens = []
for nm, g in pk_df.groupby('model'):
    sens.append(dict(model=nm,
                     RMSE_vs_table5=rmse(g.pred_peak, g.table5_peak),
                     RMSE_vs_digitised=rmse(g.pred_peak, g.digitised_peak),
                     mean_bias_vs_table5=float(np.mean(g.err_vs_table5)),
                     mean_bias_vs_digitised=float(np.mean(g.err_vs_digitised))))
S = pd.DataFrame(sens)
S['rank_table5'] = S.RMSE_vs_table5.rank()
S['rank_digitised'] = S.RMSE_vs_digitised.rank()
S = S.sort_values('RMSE_vs_digitised')
S.to_csv(os.path.join(OUT,'table5_sensitivity_summary.csv'), index=False)
print(S.round(3).to_string(index=False))
from scipy import stats as st
rho = st.spearmanr(S.RMSE_vs_table5, S.RMSE_vs_digitised)
print('Spearman rank correlation of model error across the two data sources: rho=%.3f p=%.3f' % (rho.statistic, rho.pvalue))
json.dump({'spearman_rho':float(rho.statistic),'p':float(rho.pvalue),
           'n_models':int(len(S))}, open(os.path.join(OUT,'table5_rank_corr.json'),'w'), indent=2)

# ---------------------------------------------------------------- PDP / ALE-style trends
pd_rows = []
for ns in [0,1,2,3,4,5]:
    for nm, m in fitted.items():
        sel = (G[:,0]==ns)&(G[:,1]==1.0)
        pd_rows.append(dict(axis='nS', value=ns, model=nm, peak=float((pinnG if m is None else m.predict(G[sel]))[np.argmax((pinnG if m is None else m.predict(G[sel])))])))
for pva in [0.5,1.0,1.5,2.0]:
    for nm, m in fitted.items():
        sel = (G[:,0]==2.0)&(G[:,1]==pva)
        v = pinnG[sel] if m is None else m.predict(G[sel])
        pd_rows.append(dict(axis='PVA', value=pva, model=nm, peak=float(v.max())))
PD = pd.DataFrame(pd_rows)
PD.to_csv(os.path.join(OUT,'pdp_trends.csv'), index=False)
exp_ns = [float(pr[pr.nS==n].groupby('PVA').sigma_peak.mean().max()) for n in [0,1,2,3,4,5]]
exp_pva = [float(pr[pr.PVA==p].groupby('nS').sigma_peak.mean().max()) for p in [0.5,1.0,1.5,2.0]]
json.dump({'experimental_peak_vs_nS_at_best_PVA': exp_ns,
           'experimental_peak_vs_PVA_at_best_nS': exp_pva}, open(os.path.join(OUT,'pdp_experimental.json'),'w'), indent=2)
print('experimental peak vs nS:', exp_ns)

# ---------------------------------------------------------------- GP conformal
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(); Xt = sc.fit_transform(Xtr); Xs_ = sc.transform(Xte)
gp = GaussianProcessRegressor(kernel=P.C(1.0)*P.RBF([1.,1.,1.])+P.WhiteKernel(1e-3),
                              normalize_y=True, random_state=42).fit(Xt, ytr)
mu, sd = gp.predict(Xs_, return_std=True)
picp95 = float(np.mean(np.abs(yte-mu) <= 1.96*sd)*100)
mpiw95 = float(np.mean(2*1.96*sd))
# split-conformal scaling on the calibration set (train on 80% of train, calibrate on 20%)
n = len(Xt); rs = np.random.RandomState(0); perm = rs.permutation(n); ncal = int(0.25*n)
cal_i, fit_i = perm[:ncal], perm[ncal:]
gp2 = GaussianProcessRegressor(kernel=P.C(1.0)*P.RBF([1.,1.,1.])+P.WhiteKernel(1e-3),
                               normalize_y=True, random_state=42).fit(Xt[fit_i], ytr[fit_i])
muc, sdc = gp2.predict(Xt[cal_i], return_std=True)
score = np.abs(ytr[cal_i]-muc)/np.maximum(sdc,1e-9)
q = float(np.quantile(score, 0.95))
mu3, sd3 = gp2.predict(Xs_, return_std=True)
picp_c = float(np.mean(np.abs(yte-mu3) <= q*sd3)*100)
mpiw_c = float(np.mean(2*q*sd3))
conf = dict(nominal=95.0, PICP_raw=picp95, MPIW_raw=mpiw95, conformal_q=q,
            PICP_conformal=picp_c, MPIW_conformal=mpiw_c, n_cal=ncal)
json.dump(conf, open(os.path.join(OUT,'gp_conformal.json'),'w'), indent=2)
print('GP calibration:', json.dumps(conf, indent=2))

# ---------------------------------------------------------------- hyperparameter sensitivity
hp = []
for d in [3,4,5,6]:
    m = P.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=d, subsample=0.9,
                       colsample_bytree=0.9, random_state=42, verbosity=0).fit(Xtr,ytr)
    hp.append(dict(model='XGBoost', param='max_depth', value=d, RMSE=rmse(yte,m.predict(Xte))))
for ne in [200,500,800]:
    m = P.XGBRegressor(n_estimators=ne, learning_rate=0.05, max_depth=4, subsample=0.9,
                       colsample_bytree=0.9, random_state=42, verbosity=0).fit(Xtr,ytr)
    hp.append(dict(model='XGBoost', param='n_estimators', value=ne, RMSE=rmse(yte,m.predict(Xte))))
for ne in [100,400,800]:
    m = RandomForestRegressor(n_estimators=ne, random_state=42, n_jobs=-1).fit(Xtr,ytr)
    hp.append(dict(model='Random Forest', param='n_estimators', value=ne, RMSE=rmse(yte,m.predict(Xte))))
for md in [6,8,12]:
    m = RandomForestRegressor(n_estimators=400, max_depth=md, random_state=42, n_jobs=-1).fit(Xtr,ytr)
    hp.append(dict(model='Random Forest', param='max_depth', value=md, RMSE=rmse(yte,m.predict(Xte))))
for r in [42,7,2024]:
    m = RandomForestRegressor(n_estimators=400, random_state=r, n_jobs=-1).fit(Xtr,ytr)
    hp.append(dict(model='Random Forest', param='random_state', value=r, RMSE=rmse(yte,m.predict(Xte))))
for ls in [0.5,1.0,2.0]:
    k = P.C(1.0)*P.RBF([ls,ls,ls])+P.WhiteKernel(1e-3)
    m = GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=42).fit(Xt,ytr)
    hp.append(dict(model='Gaussian Process', param='RBF length_scale', value=ls, RMSE=rmse(yte,m.predict(Xs_))))
H = pd.DataFrame(hp); H['model_param'] = H.model + ' | ' + H.param
H.to_csv(os.path.join(OUT,'hyperparam_sensitivity.csv'), index=False)
print(H.groupby(['model','param']).RMSE.agg(['min','max','std']).round(4).to_string())
print('PART 2 COMPLETE')
