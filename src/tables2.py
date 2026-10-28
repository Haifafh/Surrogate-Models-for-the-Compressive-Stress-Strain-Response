#!/usr/bin/env python3
"""Regenerate metrics tables + reviewer-analytics figure from the corrected run."""
import os, sys, json, copy, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT = os.path.expanduser('~/workspace/paper')
sys.path.insert(0, ROOT); os.chdir(ROOT)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pipeline as P, figures as F
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy import stats as st
OUT='out'

# ---------------------------------------------------------------- extended metrics
z = np.load(os.path.join(OUT,'predictions2.npz'), allow_pickle=True)
df, pr, curves = P.build_dataset()
blind = df.apply(lambda r: (r.nS, r.PVA) in P.TEST_COMBOS, axis=1)
B_te = df[blind]
coefB = P.fit_modulus_surface(pr, set(zip(df[~blind].nS, df[~blind].PVA)))
y = B_te.stress_MPa.values
grp = pd.factorize(pd.MultiIndex.from_arrays([B_te.nS.values, B_te.PVA.values]).to_flat_index())[0]
lat = {r.model.replace('_',' '): r.infer_ms for r in pd.read_csv(os.path.join(OUT,'cost_all.csv')).itertuples()}

def ext(y, yhat):
    r = yhat - y
    r2 = 1 - np.sum(r**2)/np.sum((y-y.mean())**2)
    rmse = float(np.sqrt(np.mean(r**2)))
    m = y > 1.0
    alpha = np.std(yhat)/np.std(y); beta = np.mean(yhat)/np.mean(y)
    rr = float(np.corrcoef(y, yhat)[0,1])
    return dict(R2=float(r2), RMSE=rmse, MAE=float(np.mean(np.abs(r))),
                MaxAE=float(np.max(np.abs(r))), ME=float(np.mean(r)),
                MAPE=float(np.mean(np.abs((y[m]-yhat[m])/y[m]))*100),
                NRMSE=rmse/(y.max()-y.min()), r=rr,
                rho=float(st.spearmanr(y,yhat).statistic),
                d=1-np.sum(r**2)/np.sum((np.abs(yhat-y.mean())+np.abs(y-y.mean()))**2),
                KGE=1-np.sqrt((rr-1)**2+(alpha-1)**2+(beta-1)**2))

rows=[]
for k in z.files:
    sp,nm = k.split('||')
    if sp!='B_blind_mixtures': continue
    yh = z[k]; d = ext(y, yh)
    # per-mixture peak errors
    dd = B_te.copy(); dd['p']=yh
    pe=[]; ee=[]
    for _,g in dd.groupby(['nS','PVA']):
        pe.append(abs(g.p.loc[g.p.idxmax()]-g.stress_MPa.max())/g.stress_MPa.max()*100)
        ee.append(abs(g.loc[g.p.idxmax(),'strain_um_m']-g.loc[g.stress_MPa.idxmax(),'strain_um_m'])
                  /g.loc[g.stress_MPa.idxmax(),'strain_um_m']*100)
    d.update(peak_stress_err=float(np.mean(pe)), peak_strain_err=float(np.mean(ee)),
             inference_ms=lat.get(nm.replace('_',' '), np.nan), model=nm)
    rows.append(d)
E = pd.DataFrame(rows).sort_values('R2', ascending=False)
E.to_csv(os.path.join(OUT,'metrics_extended.csv'), index=False)

# ---------------------------------------------------------------- table 4 (benchmark, both splits)
B = pd.read_csv(os.path.join(OUT,'benchmark_metrics.csv'))
piv = B.pivot_table(index='model', columns='split', values=['R2','RMSE','MAE','MAPE'])
def disp(m):
    d={'NN data-only (matched control)':'NN data-only (matched control)',
       'PINN (physics-informed)':'PINN (auto-selected, $\\lambda=0$)',
       'Polynomial Reg. (deg 3)':'Polynomial regression (deg.~3)',
       'k-NN (k=5)':'$k$-NN ($k=5$)'}
    return d.get(m, m)
L=[r'\begin{table}[t]', r'\centering',
   r'\caption{Benchmark on both evaluation protocols. Split~A is a random 80/20 partition of the '
   r'collocation points; Split~B is the blind-mixture protocol, in which five complete stress--strain '
   r'curves are withheld from training. The best value in each column is bold. The deployed '
   r'physics-informed network selected $\lambda_E=\lambda_D=\lambda_S=0$ on the inner validation split '
   r'and is therefore numerically identical to the matched data-only control; a single row is reported.',
   r'}', r'\label{tab:bench}', r'\scriptsize', r'\setlength{\tabcolsep}{3.0pt}',
   r'\begin{tabular}{@{}lcccccccc@{}}', r'\toprule',
   r'\multirow{2}{*}{Model} & \multicolumn{4}{c}{Split A --- random points} & \multicolumn{4}{c}{Split B --- blind mixtures}\\',
   r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
   r' & $R^2$ & RMSE & MAE & MAPE & $R^2$ & RMSE & MAE & MAPE\\', r'\midrule']
seen=set()
for m in E.model:
    if m in seen: continue
    seen.add(m)
    r = piv.loc[m] if m in piv.index else None
    def cell(sp, k, fmt):
        try:
            v = float(piv.loc[m, (k, sp)])
            return '---' if not np.isfinite(v) else (fmt % v)
        except Exception: return '---'
    L.append('%s & %s & %s\\\\' % (disp(m),
        ' & '.join([cell('A_random80-20','R2','%.4f'), cell('A_random80-20','RMSE','%.2f'),
                    cell('A_random80-20','MAE','%.2f'), cell('A_random80-20','MAPE','%.2f')]),
        ' & '.join([cell('B_blind_mixtures','R2','%.4f'), cell('B_blind_mixtures','RMSE','%.2f'),
                    cell('B_blind_mixtures','MAE','%.2f'), cell('B_blind_mixtures','MAPE','%.2f')])))
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
open('table4.tex','w').write('\n'.join(L))

# ---------------------------------------------------------------- table 5 (extended)
cols=['R2','RMSE','MaxAE','ME','r','d','KGE','peak_stress_err']
best={}
for c in ['R2','r','d','KGE']: best[c]=E[c].max()
for c in ['RMSE','MaxAE','peak_stress_err']: best[c]=E[c].min()
best['ME']=E.ME.abs().idxmin()
T=[r'\begin{table}[t]', r'\centering',
   r'\caption{Extended accuracy metrics on the blind-mixture split. RMSE, MaxAE and ME are in MPa; the '
   r'peak-stress error is in \%. The best value in each column is bold.}', r'\label{tab:extmetrics}',
   r'\scriptsize', r'\setlength{\tabcolsep}{3.6pt}',
   r'\begin{tabular}{@{}lcccccccc@{}}', r'\toprule',
   r'Model & $R^2$ & RMSE & MaxAE & ME & $r$ & $d$ & KGE & peak stress err.\\', r'\midrule']
for r in E.itertuples():
    nm = disp(r.model)
    if 'PINN (auto' in nm: nm = r'\textbf{%s}' % nm
    cells=[]
    for c in cols:
        v=getattr(r,c)
        s='%+.2f'%v if c=='ME' else ('%.2f'%v if c in ('RMSE','MaxAE','peak_stress_err') else '%.4f'%v)
        isb = (c=='ME' and r.Index==best['ME']) or (c!='ME' and np.isfinite(best[c])
              and abs(v-best[c])<5e-3*max(1,abs(best[c])))
        cells.append(r'\textbf{%s}'%s if isb else s)
    T.append('%s & %s\\\\' % (nm,' & '.join(cells)))
T += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
open('table5.tex','w').write('\n'.join(T))
print('metrics_extended.csv + table4.tex + table5.tex regenerated (%d models)' % len(E))
print(E[['model','R2','RMSE','MaxAE','KGE','peak_stress_err']].round(4).to_string(index=False))

# ---------------------------------------------------------------- hyperparameter sensitivity (fixed)
from xgboost import XGBRegressor
Xtr = df[~blind][['nS','PVA','strain_pct']].values; ytr = df[~blind].stress_MPa.values
Xte = B_te[['nS','PVA','strain_pct']].values
def rm(a,b): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))
sc = __import__('sklearn.preprocessing', fromlist=['StandardScaler']).StandardScaler()
Xt = sc.fit_transform(Xtr); Xq = sc.transform(Xte)
hp=[]
for d in [3,4,5,6]:
    m=XGBRegressor(n_estimators=500,learning_rate=0.05,max_depth=d,subsample=0.9,colsample_bytree=0.9,random_state=42,verbosity=0).fit(Xtr,ytr)
    hp.append(dict(model='XGBoost',param='max_depth',value=d,RMSE=rm(yte:=y,m.predict(Xte))))
for ne in [200,500,800]:
    m=XGBRegressor(n_estimators=ne,learning_rate=0.05,max_depth=4,subsample=0.9,colsample_bytree=0.9,random_state=42,verbosity=0).fit(Xtr,ytr)
    hp.append(dict(model='XGBoost',param='n_estimators',value=ne,RMSE=rm(y,m.predict(Xte))))
for ne in [100,400,800]:
    m=RandomForestRegressor(n_estimators=ne,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    hp.append(dict(model='Random Forest',param='n_estimators',value=ne,RMSE=rm(y,m.predict(Xte))))
for md in [6,8,12]:
    m=RandomForestRegressor(n_estimators=400,max_depth=md,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    hp.append(dict(model='Random Forest',param='max_depth',value=md,RMSE=rm(y,m.predict(Xte))))
for rs in [42,7,2024]:
    m=RandomForestRegressor(n_estimators=400,random_state=rs,n_jobs=-1).fit(Xtr,ytr)
    hp.append(dict(model='Random Forest',param='random_state',value=rs,RMSE=rm(y,m.predict(Xte))))
for ls in [0.5,1.0,2.0]:
    k=P.C(1.0)*P.RBF([ls,ls,ls])+P.WhiteKernel(1e-3)
    m=GaussianProcessRegressor(kernel=k,normalize_y=True,random_state=42).fit(Xt,ytr)
    hp.append(dict(model='Gaussian Process',param='RBF length_scale',value=ls,RMSE=rm(y,m.predict(Xq))))
H=pd.DataFrame(hp); H.to_csv(os.path.join(OUT,'hyperparam_sensitivity.csv'),index=False)
print(H.groupby(['model','param']).RMSE.agg(['min','max','std']).round(4).to_string())

# ---------------------------------------------------------------- Figure 13
ms=pd.read_csv(os.path.join(OUT,'multisplit.csv'))
PD=pd.read_csv(os.path.join(OUT,'pdp_trends.csv'))
ex=json.load(open(os.path.join(OUT,'pdp_experimental.json')))
fig,axes=plt.subplots(1,2,figsize=(11,4.0))
ax=axes[0]
short={'XGBoost':'XGBoost','NN data-only (matched control)':'NN (control)',
       'PINN (physics-informed)':'PINN','Gaussian Process':'GP','Random Forest':'RF',
       'Deep ANN (MLP 64-64-32)':'Deep ANN','Linear Regression':'Linear'}
order=ms.groupby('model').RMSE.mean().sort_values().index.tolist()
for i,m in enumerate(order):
    v=ms[ms.model==m].RMSE.values
    jit=np.random.RandomState(i).uniform(-.10,.10,len(v))
    ax.scatter(np.full(len(v),i)+jit,v,s=16,color=F.ABL_COLORS[i%4],alpha=.85,edgecolors='none',zorder=3)
    ax.plot([i-.28,i+.28],[np.median(v)]*2,color='#222',lw=1.8,zorder=4)
ax.set_xticks(range(len(order))); ax.set_xticklabels([short[m] for m in order],rotation=22,ha='right',fontsize=7.5)
ax.set_ylabel('blind-test RMSE (MPa)',fontsize=9); ax.set_title('(a) 15 resampled blind-mixture splits',fontsize=9)
ax.tick_params(labelsize=8); ax.grid(alpha=.25,lw=.5)
ax=axes[1]
ns=[0,1,2,3,4,5]
ax.plot(ns,ex['experimental_peak_vs_nS_at_best_PVA'],'o-',color='#111',lw=2,ms=6,label='experiment',zorder=5)
for m,k in [('PINN (physics-informed)',0),('XGBoost',1),('Gaussian Process',2),('Deep ANN (MLP 64-64-32)',3)]:
    g=PD[(PD.axis=='nS')&(PD.model==m)].sort_values('value')
    ax.plot(g.value,g.peak,'--',color=F.ABL_COLORS[k],lw=1.3,label=short[m])
ax.set_xlabel('nanosilica content (\%)',fontsize=9); ax.set_ylabel('peak stress (MPa)',fontsize=9)
ax.set_title('(b) peak stress versus $n_S$ (PVA = 1.0\%)',fontsize=9)
ax.legend(fontsize=7,frameon=False); ax.tick_params(labelsize=8); ax.grid(alpha=.25,lw=.5)
fig.tight_layout()
F.save(fig,'fig13_reviewer_analytics')
print('fig13 written')
