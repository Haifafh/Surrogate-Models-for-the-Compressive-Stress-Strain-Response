#!/usr/bin/env python3
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
P1=os.path.expanduser('~/workspace/paper'); P2=os.path.expanduser('~/workspace/paper2')
sys.path.insert(0,P1); sys.path.insert(0,P2); os.chdir(P1)
import pipeline as P
from kan import KAN
from deeponet import DeepONet
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import figures as F1                      # palette + save() + PGF backend
OUT2=os.path.join(P2,'out2'); os.makedirs(OUT2,exist_ok=True)
FIG=os.path.join(P2,'figures')
t0=time.time()
def tick(m): print('[%6.1fs] %s'%(time.time()-t0,m),flush=True)

df,pr,curves=P.build_dataset()
blind=df.apply(lambda r:(r.nS,r.PVA) in P.TEST_COMBOS,axis=1)
B_tr,B_te=df[~blind],df[blind]
coefB=P.fit_modulus_surface(pr,set(zip(B_tr.nS,B_tr.PVA)))
Xq,yq=B_te[['nS','PVA','strain_pct']].values,B_te.stress_MPa.values
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))
def r2f(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(1-np.sum((a-b)**2)/np.sum((b-b.mean())**2))
def pcl(dfx):
    return [(k,dfx[(dfx.nS==k[0])&(dfx.PVA==k[1])][['strain_pct','stress_MPa']].values)
            for k in dfx.groupby(['nS','PVA']).groups.keys()]
def Esurf(coef,X):
    return np.array([coef[0]+coef[1]*x[0]+coef[2]*x[1]+coef[3]*x[0]**2+coef[4]*x[1]**2+coef[5]*x[0]*x[1] for x in X])

def fit_kan(tr,coef,kw,Xt,seeds=(0,)):
    acc=np.zeros(len(Xt));per=[]
    for s in seeds:
        m=KAN(seed=s,**kw); m.coef=coef
        Xs=tr[['nS','PVA','strain_pct']].values.astype(float); ys=tr['stress_MPa'].values.astype(float)
        m.fit(Xs,ys,Esurf(coef,Xs),pcl(tr)); p=m.predict(Xt); acc+=p; per.append(p)
    return acc/len(seeds),np.array(per)
def fit_don(tr,coef,kw,Xt,seeds=(0,)):
    acc=np.zeros(len(Xt));per=[]
    for s in seeds:
        m=DeepONet(seed=s,**kw); m.coef=coef
        Xs=tr[['nS','PVA','strain_pct']].values.astype(float); ys=tr['stress_MPa'].values.astype(float)
        m.fit(Xs,ys); p=m.predict(Xt); acc+=p; per.append(p)
    return acc/len(seeds),np.array(per)

KAN_KW=dict(dims=(3,12,1),G=8,p=3,lr=5e-3,epochs=3000,batch=64)
KAN_PH=dict(KAN_KW,lam_E=3.0)
DON_KW=dict(p=16,lr=5e-3,epochs=3000,batch=128)
DON_PH=dict(DON_KW,lam_E=3.0)
OPT0=dict(lr=0.01,batch=64,epochs=12000,lam_E=0.0,lam_dmg=0.0,lam_sm=0.0)
OPT1=dict(lr=0.01,batch=64,epochs=12000,lam_E=1.0,lam_dmg=1.0,lam_sm=1.0)

tick('blind-split models')
res={}
k1,_=fit_kan(B_tr,coefB,KAN_KW,Xq); res['PIKAN (data only)']=(k1,KAN_KW,'kan')
tick(' PIKAN data %.4f/%.4f'%(r2f(yq,k1),rmse(yq,k1)))
k2,_=fit_kan(B_tr,coefB,KAN_PH,Xq); res['PIKAN (elastic pin on)']=(k2,KAN_PH,'kan')
tick(' PIKAN phys %.4f/%.4f'%(r2f(yq,k2),rmse(yq,k2)))
d1,_=fit_don(B_tr,coefB,DON_KW,Xq); res['DeepONet']=(d1,DON_KW,'don')
tick(' DON %.4f/%.4f'%(r2f(yq,d1),rmse(yq,d1)))
d2,_=fit_don(B_tr,coefB,DON_PH,Xq); res['PINO (elastic pin on)']=(d2,DON_PH,'don')
tick(' PINO %.4f/%.4f'%(r2f(yq,d2),rmse(yq,d2)))
for tag,o in [('MLP data-only ($\\lambda=0$)',OPT0),('MLP-PINN ($\\lambda=1,1,1$)',OPT1)]:
    acc=np.zeros(len(Xq))
    for s in (0,1):
        m=P.train_pinn(B_tr,pr,coefB,seed=s,**o); acc+=m.predict(Xq,m.scX,m.scY)
    acc/=2.0; res[tag]=(acc,o,'mlp'); tick(' %s %.4f/%.4f'%(tag,r2f(yq,acc),rmse(yq,acc)))
from xgboost import XGBRegressor
xg=XGBRegressor(n_estimators=500,learning_rate=0.05,max_depth=4,subsample=0.9,
                colsample_bytree=0.9,random_state=42,verbosity=0).fit(
                B_tr[['nS','PVA','strain_pct']].values,B_tr.stress_MPa.values)
res['XGBoost']=(xg.predict(Xq),{},'xgb')

rows=[];per_store={}
for nm,(p_,kw_,kind) in res.items():
    dg=P.physical_diagnostics(B_te,p_,coefB); m=P.metrics(yq,p_)
    npar={'kan':KAN(dims=kw_.get('dims',(3,12,1)),G=kw_.get('G',8),p=kw_.get('p',3)).n_params(),
          'don':DeepONet(p=kw_.get('p',16)).n_params(),
          'mlp':6368,'xgb':None}.get(kind)
    rows.append(dict(model=nm,R2=m['R2'],RMSE=m['RMSE'],MAE=m['MAE'],MAPE=m['MAPE'],
                     params=npar,lam_E=kw_.get('lam_E',0.0),**dg))
    per_store[nm]=p_
R=pd.DataFrame(rows).sort_values('RMSE'); R.to_csv(os.path.join(OUT2,'blind_results.csv'),index=False)
tick('blind_results.csv'); print(R.round(4).to_string(index=False))
np.savez_compressed(os.path.join(OUT2,'preds2.npz'),**per_store,Xq=Xq,yq=yq)

# ---------- multi-split (6) ----------
NS=6; combos=sorted(set(zip(df.nS,df.PVA))); rng=np.random.RandomState(2024); ms=[]
tick('multi-split %d (1 seed each)'%NS)
for it in range(NS):
    hold=set(combos[i] for i in rng.choice(len(combos),len(P.TEST_COMBOS),replace=False))
    isb=df.apply(lambda r:(r.nS,r.PVA) in hold,axis=1); tr,te=df[~isb],df[isb]
    Xt=te[['nS','PVA','strain_pct']].values; yt=te.stress_MPa.values
    cf=P.fit_modulus_surface(pr,set(zip(tr.nS,tr.PVA)))
    for tag,fn,kw in [('PIKAN',fit_kan,KAN_KW),('DeepONet',fit_don,DON_KW)]:
        p_,_=fn(tr,cf,kw,Xt); ms.append(dict(it=it,model=tag,R2=r2f(yt,p_),RMSE=rmse(yt,p_)))
    acc=np.zeros(len(Xt))
    for s in (0,):
        m=P.train_pinn(tr,pr,cf,seed=s,**OPT0); acc+=m.predict(Xt,m.scX,m.scY)
    ms.append(dict(it=it,model='MLP data-only',R2=r2f(yt,acc),RMSE=rmse(yt,acc)))
    x2=XGBRegressor(n_estimators=500,learning_rate=0.05,max_depth=4,subsample=0.9,
                    colsample_bytree=0.9,random_state=42,verbosity=0).fit(
                    tr[['nS','PVA','strain_pct']].values,tr.stress_MPa.values)
    pp=x2.predict(Xt); ms.append(dict(it=it,model='XGBoost',R2=r2f(yt,pp),RMSE=rmse(yt,pp)))
    tick('  split %d/%d'%(it+1,NS))
M=pd.DataFrame(ms); M.to_csv(os.path.join(OUT2,'multisplit_new.csv'),index=False)
S=M.groupby('model').agg(R2_mean=('R2','mean'),R2_sd=('R2','std'),RMSE_mean=('RMSE','mean'),
                         RMSE_sd=('RMSE','std'),n=('R2','size')).sort_values('RMSE_mean')
S.to_csv(os.path.join(OUT2,'multisplit_new_summary.csv')); print(S.round(4).to_string())
from scipy import stats as st
W={}
for m in ['MLP data-only','DeepONet','XGBoost']:
    a=M[M.model=='PIKAN'].sort_values('it').RMSE.values; b=M[M.model==m].sort_values('it').RMSE.values
    if len(a)==len(b)>4:
        w=st.wilcoxon(a,b); W['PIKAN_vs_%s'%m]={'p':float(w.pvalue),'median_diff':float(np.median(a-b)),'n':int(len(a))}
json.dump(W,open(os.path.join(OUT2,'significance_new.json'),'w'),indent=2)
tick('significance'); print(json.dumps(W,indent=2))
tick('ANALYSIS DONE')
