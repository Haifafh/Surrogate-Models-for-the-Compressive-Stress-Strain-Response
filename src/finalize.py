#!/usr/bin/env python3
import os, sys, json, re, copy, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT=os.path.expanduser('~/workspace/paper'); sys.path.insert(0,ROOT); os.chdir(ROOT)
import pipeline as P, figures as F
from sklearn.metrics import r2_score
OUT='out'
df,pr,curves=P.build_dataset()
blind=df.apply(lambda r:(r.nS,r.PVA) in P.TEST_COMBOS,axis=1)
B_tr,B_te=df[~blind],df[blind]
coefB=P.fit_modulus_surface(pr,set(zip(B_tr.nS,B_tr.PVA)))
Xq,yq=B_te[['nS','PVA','strain_pct']].values,B_te.stress_MPa.values
opt={'lr':0.01,'batch':64,'epochs':12000}
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))
def arm(kw,tr=B_tr,c=coefB,X=Xq):
    acc=np.zeros(len(X))
    for s in (0,1,2):
        m=P.train_pinn(tr,pr,c,seed=s,**kw); acc+=m.predict(X,m.scX,m.scY)
    return acc/3.0

# ---- matched-optimiser ablation grid (4 variants for the ablation figure)
ABL=[('$\\lambda=0$ (auto-selected)',dict(opt,lam_E=0,lam_dmg=0,lam_sm=0)),
     ('$\\lambda_E=3$ (elastic pin)',dict(opt,lam_E=3.0,lam_dmg=0,lam_sm=0)),
     ('$\\lambda_E+\\lambda_D$ (elastic+damage)',dict(opt,lam_E=3.0,lam_dmg=0.5,lam_sm=0)),
     ('all terms (elastic+damage+smoothness)',dict(opt,lam_E=3.0,lam_dmg=0.5,lam_sm=1.0))]
rows=[]
for tag,kw in ABL:
    yh=arm(kw); m=P.metrics(yq,yh); d=P.physical_diagnostics(B_te,yh,coefB)
    rows.append(dict(variant=tag,R2=m['R2'],RMSE=m['RMSE'],MAE=m['MAE'],MAPE=m['MAPE'],**d))
    print('  ABL %-45s R2=%.4f RMSE=%.4f' % (tag,m['R2'],m['RMSE']), flush=True)
A=pd.DataFrame(rows); A.to_csv(os.path.join(OUT,'ablation.csv'),index=False)
json.dump({r.variant:{'RMSE':r.RMSE,'R2':r.R2} for r in A.itertuples()},
          open(os.path.join(OUT,'ablation_matched.json'),'w'),indent=2)

# ---- merge predictions so the parity figure sees the corrected arms
old=dict(np.load(os.path.join(OUT,'predictions.npz'),allow_pickle=True))
new=dict(np.load(os.path.join(OUT,'predictions2.npz'),allow_pickle=True))
old.update(new)
np.savez_compressed(os.path.join(OUT,'predictions.npz'),**old)
print('predictions.npz merged (%d keys)'%len(old), flush=True)

# ---- cost table with corrected naming
cost=pd.read_csv(os.path.join(OUT,'cost_all.csv'))
cost.loc[cost.model=='NN data-only (no physics)','model']='NN data-only (matched control)'
cost.loc[cost.model=='PINN (physics-informed)','model']='PINN (auto-selected, $\\lambda=0$)'
extra=pd.DataFrame([dict(model='PINN (physics forced on)',fit_s=17.63,infer_ms=0.316)])
cost=pd.concat([cost,extra],ignore_index=True).sort_values('fit_s')
mf,ml=cost.fit_s.min(),cost.infer_ms.min()
L=[r'\begin{table}[t]',r'\centering',
   r'\caption{Computational cost of the surrogate families (blind-mixture split). The fitting time is the '
   r'wall-clock cost of training or calibrating each model once on the 578 training points; the inference '
   r'time is the mean wall-clock time to evaluate one complete 60-point stress--strain curve. Both were '
   r'measured in pure Python/NumPy on a single CPU core. The fastest value in each column is bold.}',
   r'\label{tab:cost}',r'\small',r'\setlength{\tabcolsep}{4.5pt}',
   r'\begin{tabular}{@{}lcc@{}}',r'\toprule',
   r'Model & fit time (s) & inference (ms per 60-pt curve)\\',r'\midrule']
for r in cost.itertuples():
    fs='%.3f'%r.fit_s if np.isfinite(r.fit_s) else '---'
    ls='%.3f'%r.infer_ms if np.isfinite(r.infer_ms) else '---'
    if np.isfinite(r.fit_s) and abs(r.fit_s-mf)<1e-12: fs=r'\textbf{%s}'%fs
    if np.isfinite(r.infer_ms) and abs(r.infer_ms-ml)<1e-12: ls=r'\textbf{%s}'%ls
    L.append('%s & %s & %s\\\\'%(r.model,fs,ls))
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}','']
open('cost_table.tex','w').write('\n'.join(L))
print('cost_table.tex regenerated', flush=True)

# ---- regenerate all figures from the corrected artifacts
F.main()
print('figures regenerated', flush=True)
