#!/usr/bin/env python3
"""Reviewer-response analysis (part 1).
P0: config manifest + TRUE ablation control (matched optimiser, only lambda differ)
P1: multi-split blind evaluation, cluster bootstrap + Wilcoxon, inter-seed variance
P2: Table-5 peak sensitivity
"""
import os, sys, json, time, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT = os.path.expanduser('~/workspace/paper')
sys.path.insert(0, ROOT); os.chdir(ROOT)
import pipeline as P
from sklearn.metrics import r2_score
from scipy import stats

OUT = 'out'
t_start = time.time()
def tick(msg): print('[%6.1fs] %s' % (time.time()-t_start, msg), flush=True)

df, pr, curves = P.build_dataset()
tick('dataset %s | columns %s' % (df.shape, list(df.columns)))

blind = df.apply(lambda r: (r.nS, r.PVA) in P.TEST_COMBOS, axis=1)
B_tr, B_te = df[~blind], df[blind]
A_idx = np.random.RandomState(42).permutation(len(df)); ntr = int(0.8*len(df))
A_tr, A_te = df.iloc[A_idx[:ntr]], df.iloc[A_idx[ntr:]]
coefB = P.fit_modulus_surface(pr, set(zip(B_tr.nS, B_tr.PVA)))
coefA = P.fit_modulus_surface(pr, set(zip(A_tr.nS, A_tr.PVA)))
XteB, yteB = B_te[['nS','PVA','strain_pct']].values, B_te.stress_MPa.values
tick('splits: A %d/%d  B %d/%d' % (len(A_tr), len(A_te), len(B_tr), len(B_te)))

def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a-b)**2)))
def r2f(a, b): return float(r2_score(a, b))

def arm(df_tr, coef, kw, Xq, seeds=(0,1,2)):
    """Train a PINN arm as a multi-seed ensemble; return (ensemble pred, per-seed preds)."""
    per = []
    for s in seeds:
        m = P.train_pinn(df_tr, pr, coef, seed=s, **kw)
        per.append(m.predict(Xq, m.scX, m.scY))
    per = np.array(per)
    return per.mean(0), per

# ============================================================ P0 config manifest
tick('calibrating PINN on inner-validation split (physics weights selected on validation only)')
calib = P.calibrate_pinn(B_tr, pr, coefB, verbose=True)
opt_keys = {k: calib[k] for k in ('lr','batch','epochs') if k in calib}
ctr_kw   = dict(opt_keys, lam_E=0.0, lam_dmg=0.0, lam_sm=0.0)
pin_kw   = dict(opt_keys, lam_E=calib.get('lam_E',1.0), lam_dmg=calib.get('lam_dmg',1.0),
                lam_sm=calib.get('lam_sm',1.0))
tick('selected config: %s' % json.dumps(calib))

# arm C: physics FORCED on with the best NON-ZERO grid point (what if physics is switched on?)
nonzero = [g for g in P.PINN_GRID if any(v > 0 for v in g.values())]
force_kw = dict(opt_keys, **nonzero[0])
manifest = {'calibrated_config': calib, 'matched_optimiser': opt_keys,
            'arm_deployed': pin_kw, 'arm_control': ctr_kw, 'arm_physics_forced': force_kw,
            'seeds': [0,1,2], 'split_B_train_mixtures': sorted(set(zip(B_tr.nS,B_tr.PVA))),
            'split_B_test_mixtures': sorted(set(zip(B_te.nS,B_te.PVA))),
            'lambdas_deployed': [pin_kw['lam_E'], pin_kw['lam_dmg'], pin_kw['lam_sm']]}
json.dump(manifest, open(os.path.join(OUT,'config_manifest.json'),'w'), indent=2, default=str)
tick('manifest written: deployed lambdas = %s' % [pin_kw['lam_E'],pin_kw['lam_dmg'],pin_kw['lam_sm']])

# ============================================================ P0 true ablation (Split B)
tick('training ablation arms on Split B (matched optimiser)')
arms = {}
for tag, kw in [('PINN (physics-informed)', pin_kw),
                ('NN data-only (matched control)', ctr_kw),
                ('PINN (physics forced on)', force_kw)]:
    pred, per = arm(B_tr, coefB, kw, XteB)
    arms[tag] = (pred, per, kw)
    tick('  %-32s R2=%.4f RMSE=%.4f' % (tag, r2f(yteB,pred), rmse(yteB,pred)))

abl = []
for tag, (pred, per, kw) in arms.items():
    d = P.physical_diagnostics(B_te, pred, coefB)
    m = P.metrics(yteB, pred)
    abl.append(dict(arm=tag, R2=m['R2'], RMSE=m['RMSE'], MAE=m['MAE'], MAPE=m['MAPE'],
                    lam_E=kw['lam_E'], lam_dmg=kw['lam_dmg'], lam_sm=kw['lam_sm'],
                    lr=kw['lr'], batch=kw['batch'], epochs=kw['epochs'],
                    seed_mean_RMSE=float(np.mean([rmse(yteB,p) for p in per])),
                    seed_sd_RMSE=float(np.std([rmse(yteB,p) for p in per])),
                    **d))
pd.DataFrame(abl).to_csv(os.path.join(OUT,'ablation_true.csv'), index=False)
tick('ablation_true.csv written')

# ============================================================ P1 inter-seed variance
sv = []
for tag,(pred,per,kw) in arms.items():
    for i,p in enumerate(per):
        sv.append(dict(arm=tag, seed=i, R2=r2f(yteB,p), RMSE=rmse(yteB,p),
                       damage_violations=P.physical_diagnostics(B_te,p,coefB)['dmg_violations']))
pd.DataFrame(sv).to_csv(os.path.join(OUT,'seed_variance.csv'), index=False)
tick('seed_variance.csv written')

# ============================================================ baseline zoo on both splits
tick('fitting the baseline zoo on both splits')
Z = P.model_zoo()
rows, store = [], {}
for sp,(tr,te,coef) in [('A_random80-20',(A_tr,A_te,coefA)), ('B_blind_mixtures',(B_tr,B_te,coefB))]:
    Xtr, ytr = tr[['nS','PVA','strain_pct']].values, tr.stress_MPa.values
    Xq,  yq  = te[['nS','PVA','strain_pct']].values, te.stress_MPa.values
    for name, mdl in Z.items():
        mdl.fit(Xtr, ytr); yh = mdl.predict(Xq)
        m = P.metrics(yq, yh); m.update(model=name, split=sp, n_train=len(tr), n_test=len(te))
        rows.append(m); store[(sp,name)] = yh
    # physics-only Carreira-Chu
    yh = np.full(len(te), np.nan); pos = {ix:i for i,ix in enumerate(te.index)}
    for k,g in te.groupby(['nS','PVA']):
        r0 = pr[(pr.nS==k[0])&(pr.PVA==k[1])].iloc[0]
        e,s = P.carreira_chu(r0.sigma_peak, r0.strain_peak_pct/100.0, r0.E0_GPa*1000.0)
        yh[[pos[i] for i in g.index]] = np.interp(g.strain_pct.values/100.0, e, s)
    m = P.metrics(yq, yh); m.update(model='Carreira-Chu (physics only)', split=sp, n_train=0, n_test=len(te))
    rows.append(m); store[(sp,'Carreira-Chu (physics only)')] = yh
    # PINN arms from above (Split B only reused; retrain for Split A)
    if sp == 'B_blind_mixtures':
        for tag,(pred,per,kw) in arms.items():
            m = P.metrics(yq, pred); m.update(model=tag, split=sp, n_train=len(tr), n_test=len(te))
            rows.append(m); store[(sp,tag)] = pred
    else:
        for tag,kw in [('PINN (physics-informed)',pin_kw), ('NN data-only (matched control)',ctr_kw)]:
            pred,_ = arm(tr, coef, kw, Xq)
            m = P.metrics(yq, pred); m.update(model=tag, split=sp, n_train=len(tr), n_test=len(te))
            rows.append(m); store[(sp,tag)] = pred
    tick('  split %s done' % sp)

res = pd.DataFrame(rows)[['split','model','n_train','n_test','R2','RMSE','MAE','MAPE']]
res.to_csv(os.path.join(OUT,'benchmark_metrics.csv'), index=False)
tick('benchmark_metrics.csv regenerated (%d rows)' % len(res))

diag = []
for (sp,nm),yh in store.items():
    if sp != 'B_blind_mixtures': continue
    d = P.physical_diagnostics(B_te, yh, coefB); d['model'] = nm; diag.append(d)
pd.DataFrame(diag).to_csv(os.path.join(OUT,'physical_diagnostics.csv'), index=False)
tick('physical_diagnostics.csv regenerated')

# ============================================================ P1 multi-split
NS = 15
combos = sorted(set(zip(df.nS, df.PVA)))
rng = np.random.RandomState(2024)
ms_rows = []
tick('multi-split blind evaluation: %d resampled %d-mixture hold-outs' % (NS, len(P.TEST_COMBOS)))
fast = [('XGBoost', Z['XGBoost']), ('Gaussian Process', Z['Gaussian Process']),
        ('Random Forest', Z['Random Forest']), ('Deep ANN (MLP 64-64-32)', Z['Deep ANN (MLP 64-64-32)']),
        ('Linear Regression', Z['Linear Regression'])]
for it in range(NS):
    hold = set(combos[i] for i in rng.choice(len(combos), len(P.TEST_COMBOS), replace=False))
    isb = df.apply(lambda r: (r.nS,r.PVA) in hold, axis=1)
    tr, te = df[~isb], df[isb]
    Xtr, ytr = tr[['nS','PVA','strain_pct']].values, tr.stress_MPa.values
    Xq, yq = te[['nS','PVA','strain_pct']].values, te.stress_MPa.values
    cf = P.fit_modulus_surface(pr, set(zip(tr.nS,tr.PVA)))
    for tag,kw in [('PINN (physics-informed)',pin_kw), ('NN data-only (matched control)',ctr_kw)]:
        pred,_ = arm(tr, cf, kw, Xq)
        ms_rows.append(dict(it=it, model=tag, R2=r2f(yq,pred), RMSE=rmse(yq,pred)))
    for name, mdl in fast:
        import copy; mdl.fit(Xtr,ytr); yh = mdl.predict(Xq)
        ms_rows.append(dict(it=it, model=name, R2=r2f(yq,yh), RMSE=rmse(yq,yh)))
    tick('  multi-split %2d/%d done' % (it+1, NS))
ms = pd.DataFrame(ms_rows); ms.to_csv(os.path.join(OUT,'multisplit.csv'), index=False)

summ = ms.groupby('model').agg(R2_mean=('R2','mean'), R2_sd=('R2','std'),
                               RMSE_mean=('RMSE','mean'), RMSE_sd=('RMSE','std'),
                               n=('R2','size')).sort_values('RMSE_mean')
summ.to_csv(os.path.join(OUT,'multisplit_summary.csv'))
tick('multisplit_summary.csv written'); print(summ.round(4).to_string())

# ============================================================ P1 significance
sig = {}
def cluster_boot(a, b, y, clusters, nboot=5000, seed=0):
    """Paired cluster bootstrap over held-out mixtures: CI for the metric difference."""
    r = np.random.RandomState(seed); idx = np.arange(len(clusters)); out = []
    for _ in range(nboot):
        pick = r.choice(idx, len(idx), replace=True)
        sel = np.concatenate([np.where(clusters == c)[0] for c in pick])
        out.append(rmse(y[sel], a[sel]) - rmse(y[sel], b[sel]))
    return float(np.percentile(out,2.5)), float(np.percentile(out,97.5))

name_a, name_b = 'PINN (physics-informed)', 'NN data-only (matched control)'
pa, pb = store[('B_blind_mixtures',name_a)], store[('B_blind_mixtures',name_b)]
cl = pd.factorize(pd.MultiIndex.from_arrays([B_te.nS.values, B_te.PVA.values]).to_flat_index())[0]
lo, hi = cluster_boot(pa, pb, yteB, cl)
sig['splitB_PINN_vs_control_RMSE_diff_CI95'] = [lo, hi]
sig['splitB_PINN_vs_control_note'] = 'cluster bootstrap over the 5 held-out mixtures'
# Wilcoxon on paired per-split RMSE (multi-split)
for m in ['NN data-only (matched control)','XGBoost','Gaussian Process','Random Forest']:
    a = ms[ms.model == name_a].sort_values('it').RMSE.values
    b = ms[ms.model == m].sort_values('it').RMSE.values
    if len(a) == len(b) and len(a) > 5:
        w = stats.wilcoxon(a, b)
        sig['multisplit_wilcoxon_PINN_vs_%s' % m] = {'stat': float(w.statistic), 'p': float(w.pvalue),
                                                     'median_RMSE_diff': float(np.median(a-b)), 'n': int(len(a))}
json.dump(sig, open(os.path.join(OUT,'significance.json'),'w'), indent=2, default=str)
tick('significance.json written'); print(json.dumps(sig, indent=2, default=str)[:1200])

np.savez_compressed(os.path.join(OUT,'predictions2.npz'),
                    **{'%s||%s' % (sp,nm): v for (sp,nm),v in store.items()})
tick('predictions2.npz written')
tick('PART 1 COMPLETE')
