#!/usr/bin/env python3
import os, sys, json, time, warnings, copy
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
P1=os.path.expanduser('~/workspace/paper'); P2=os.path.expanduser('~/workspace/paper2')
sys.path.insert(0, P1); sys.path.insert(0, P2); os.chdir(P1)
import pipeline as P
from kan import KAN
from deeponet import DeepONet
OUT2=os.path.join(P2,'out2'); os.makedirs(OUT2, exist_ok=True)
t0=time.time()
def tick(m): print('[%6.1fs] %s'%(time.time()-t0,m), flush=True)

df,pr,curves=P.build_dataset()
blind=df.apply(lambda r:(r.nS,r.PVA) in P.TEST_COMBOS, axis=1)
B_tr,B_te=df[~blind],df[blind]
coefB=P.fit_modulus_surface(pr,set(zip(B_tr.nS,B_tr.PVA)))
Xq,yq=B_te[['nS','PVA','strain_pct']].values,B_te.stress_MPa.values
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))
def r2f(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(1-np.sum((a-b)**2)/np.sum((b-b.mean())**2))
def pc(dfx):
    return [(k, dfx[(dfx.nS==k[0])&(dfx.PVA==k[1])][['strain_pct','stress_MPa']].values)
            for k in dfx.groupby(['nS','PVA']).groups.keys()]

def train_kan(tr, coef, kw, seeds=(0,1)):
    acc=np.zeros(len(Xq)); per=[]
    for s in seeds:
        m=KAN(seed=s, **kw); m.coef=coef
        Xs=tr[['nS','PVA','strain_pct']].values.astype(float)
        ys=tr['stress_MPa'].values.astype(float)
        Et=[coef[0]+coef[1]*x[0]+coef[2]*x[1]+coef[3]*x[0]**2+coef[4]*x[1]**2+coef[5]*x[0]*x[1] for x in Xs]
        m.fit(Xs, ys, Et, pc(tr))
        p=m.predict(Xq); acc+=p; per.append(p)
    return acc/len(seeds), np.array(per)

def train_don(tr, coef, kw, seeds=(0,1)):
    acc=np.zeros(len(Xq)); per=[]
    for s in seeds:
        m=DeepONet(seed=s, **kw); m.coef=coef
        Xs=tr[['nS','PVA','strain_pct']].values.astype(float)
        ys=tr['stress_MPa'].values.astype(float)
        m.fit(Xs, ys)
        p=m.predict(Xq); acc+=p; per.append(p)
    return acc/len(seeds), np.array(per)

KAN_KW=dict(dims=(3,12,1), G=8, p=3, lr=5e-3, epochs=6000, batch=64)
KAN_PHYS=dict(KAN_KW, lam_E=3.0)
DON_KW=dict(p=24, lr=5e-3, epochs=6000, batch=128)
DON_PHYS=dict(DON_KW, lam_E=3.0)

res={}
tick('training KAN (data-only) on blind split')
k1,per1=train_kan(B_tr,coefB,KAN_KW); res['PIKAN (data only)']=(k1,per1,KAN_KW)
tick('  R2=%.4f RMSE=%.4f'%(r2f(yq,k1),rmse(yq,k1)))
tick('training KAN + elastic pin (physics)')
k2,per2=train_kan(B_tr,coefB,KAN_PHYS); res['PIKAN (with physics)']=(k2,per2,KAN_PHYS)
tick('  R2=%.4f RMSE=%.4f'%(r2f(yq,k2),rmse(yq,k2)))
tick('training DeepONet')
d1,perd1=train_don(B_tr,coefB,DON_KW); res['DeepONet']=(d1,perd1,DON_KW)
tick('  R2=%.4f RMSE=%.4f'%(r2f(yq,d1),rmse(yq,d1)))
tick('training DeepONet + elastic pin (PINO)')
d2,perd2=train_don(B_tr,coefB,DON_PHYS); res['PINO (physics-informed operator)']=(d2,perd2,DON_PHYS)
tick('  R2=%.4f RMSE=%.4f'%(r2f(yq,d2),rmse(yq,d2)))

# MLP-PINN reference from paper 1
kw1=json.load(open(os.path.join(P1,'out','config_manifest.json')))['arm_deployed']
opt={k:kw1[k] for k in ('lr','batch','epochs')}
mlp=np.zeros(len(Xq)); perm=[]
for s in (0,1):
    m=P.train_pinn(B_tr,pr,coefB,seed=s,**opt); p=m.predict(Xq,m.scX,m.scY); mlp+=p; perm.append(p)
mlp/=2.0; res['MLP-PINN (paper I)']=(mlp,np.array(perm),opt)
tick('  MLP-PINN R2=%.4f RMSE=%.4f'%(r2f(yq,mlp),rmse(yq,mlp)))

from xgboost import XGBRegressor
xg=XGBRegressor(n_estimators=500,learning_rate=0.05,max_depth=4,subsample=0.9,
                colsample_bytree=0.9,random_state=42,verbosity=0).fit(
                B_tr[['nS','PVA','strain_pct']].values,B_tr.stress_MPa.values)
res['XGBoost']=(xg.predict(Xq),None,None)
tick('  XGBoost R2=%.4f RMSE=%.4f'%(r2f(yq,res['XGBoost'][0]),rmse(yq,res['XGBoost'][0])))

rows=[]
for nm,(p_,per_,kw_) in res.items():
    d=P.physical_diagnostics(B_te,p_,coefB); m=P.metrics(yq,p_)
    rows.append(dict(model=nm,R2=m['R2'],RMSE=m['RMSE'],MAE=m['MAE'],MAPE=m['MAPE'],
                     seed_sd_RMSE=(float(np.std([rmse(yq,x) for x in per_])) if per_ is not None else np.nan),
                     **d))
R=pd.DataFrame(rows).sort_values('RMSE'); R.to_csv(os.path.join(OUT2,'blind_results.csv'),index=False)
tick('blind_results.csv'); print(R.round(4).to_string(index=False))

# ---- multi-split (10) for the new methods
NS=10; combos=sorted(set(zip(df.nS,df.PVA))); rng=np.random.RandomState(2024); ms=[]
tick('multi-split %d'%NS)
for it in range(NS):
    hold=set(combos[i] for i in rng.choice(len(combos),len(P.TEST_COMBOS),replace=False))
    isb=df.apply(lambda r:(r.nS,r.PVA) in hold,axis=1); tr,te=df[~isb],df[isb]
    Xt=te[['nS','PVA','strain_pct']].values; yt=te.stress_MPa.values
    cf=P.fit_modulus_surface(pr,set(zip(tr.nS,tr.PVA)))
    for tag,fn,kw in [('PIKAN (data only)',train_kan,KAN_KW),
                      ('PIKAN (with physics)',train_kan,KAN_PHYS),
                      ('DeepONet',train_don,DON_KW),
                      ('PINO',train_don,DON_PHYS)]:
        globals()['Xq']=Xt
        try:
            p_,_=fn(tr,cf,kw,seeds=(0,)); ms.append(dict(it=it,model=tag,R2=r2f(yt,p_),RMSE=rmse(yt,p_)))
        except Exception as e: print('  fail',tag,type(e).__name__,e)
    globals()['Xq']=Xq
    tick('  split %d/%d'%(it+1,NS))
M=pd.DataFrame(ms); M.to_csv(os.path.join(OUT2,'multisplit_new.csv'),index=False)
S=M.groupby('model').agg(R2_mean=('R2','mean'),R2_sd=('R2','std'),RMSE_mean=('RMSE','mean'),
                         RMSE_sd=('RMSE','std'),n=('R2','size')).sort_values('RMSE_mean')
S.to_csv(os.path.join(OUT2,'multisplit_new_summary.csv')); print(S.round(4).to_string())
np.savez_compressed(os.path.join(OUT2,'preds2.npz'),
    **{nm:p_[0] if isinstance(p_,tuple) else p_ for nm,p_ in res.items()},
    Xq=Xq, yq=yq)
tick('DONE')
