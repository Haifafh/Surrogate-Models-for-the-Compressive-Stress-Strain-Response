#!/usr/bin/env python3
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
P1=os.path.expanduser('~/workspace/paper'); P2=os.path.expanduser('~/workspace/paper2')
sys.path.insert(0,P1); sys.path.insert(0,P2); os.chdir(P1)
import pipeline as P, figures as F1
from kan import KAN, bspline, knot_vector
from deeponet import DeepONet
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT2=os.path.join(P2,'out2')
C=F1.ABL_COLORS
def save(fig,name,d):
    F1.FIG=d; F1.save(fig,name)
D2=os.path.join(P2,'figures'); D3=os.path.join(P2,'..','paper3','figures')
D3=os.path.abspath(D3)

R=pd.read_csv(os.path.join(OUT2,'blind_results.csv'))
S=pd.read_csv(os.path.join(OUT2,'multisplit_new_summary.csv'))
M=pd.read_csv(os.path.join(OUT2,'multisplit_new.csv'))
z=np.load(os.path.join(OUT2,'preds2.npz'),allow_pickle=True)
Xq,yq=z['Xq'],z['yq']
print('=== BLIND RESULTS ==='); print(R.round(4).to_string(index=False))
print('=== MULTISPLIT ==='); print(S.round(4).to_string(index=False))
print('=== SIG ==='); print(open(os.path.join(OUT2,'significance_new.json')).read())

df,pr,curves=P.build_dataset()
blind=df.apply(lambda r:(r.nS,r.PVA) in P.TEST_COMBOS,axis=1)
B_tr,B_te=df[~blind],df[blind]
coefB=P.fit_modulus_surface(pr,set(zip(B_tr.nS,B_tr.PVA)))
Xs=B_tr[['nS','PVA','strain_pct']].values.astype(float); ys=B_tr['stress_MPa'].values.astype(float)
Es=np.array([coefB[0]+coefB[1]*x[0]+coefB[2]*x[1]+coefB[3]*x[0]**2+coefB[4]*x[1]**2+coefB[5]*x[0]*x[1] for x in Xs])
pc=[(k,B_tr[(B_tr.nS==k[0])&(B_tr.PVA==k[1])][['strain_pct','stress_MPa']].values) for k in B_tr.groupby(['nS','PVA']).groups.keys()]
print('retraining one KAN + one DeepONet for architecture figures...',flush=True)
kan=KAN(seed=0,dims=(3,12,1),G=8,p=3,lr=5e-3,epochs=3000,batch=64); kan.coef=coefB
kan.fit(Xs,ys,Es,pc)
don=DeepONet(seed=0,p=16,lr=5e-3,epochs=3000,batch=128); don.coef=coefB
don.fit(Xs,ys)
print('  done',flush=True)

# ---------------- P2 Fig 1: B-spline basis (local support) ----------------
t=knot_vector(-3,3,8,3); xx=np.linspace(-3,3,600); B=bspline(xx,t,3,0)
fig,axes=plt.subplots(1,2,figsize=(10.5,3.4))
ax=axes[0]
for i in range(B.shape[1]):
    ax.plot(xx,B[:,i],color=C[i%4],lw=1.3,alpha=.9)
ax.set_title('(a) B-spline basis, $G=8$, order 3',fontsize=9)
ax.set_xlabel('standardised input',fontsize=9); ax.set_ylabel('$B_k(x)$',fontsize=9)
ax.tick_params(labelsize=8); ax.grid(alpha=.25,lw=.5)
ax=axes[1]
ax.plot(xx,B.sum(1),color='#111',lw=1.6)
ax.set_title('(b) partition of unity',fontsize=9); ax.set_ylim(0.9,1.1)
ax.set_xlabel('standardised input',fontsize=9); ax.set_ylabel('$\\sum_k B_k(x)$',fontsize=9)
ax.tick_params(labelsize=8); ax.grid(alpha=.25,lw=.5)
fig.tight_layout(); save(fig,'p2fig01_bspline',D2)

# ---------------- P2 Fig 2: learned edge activations ----------------
L1=kan.layers[0]
xs=np.linspace(-2.5,2.5,300); Bx=bspline(xs,L1.t,L1.p,0)
fig,axes=plt.subplots(1,3,figsize=(11,3.1),sharex=True)
for k,lab in enumerate(['$n_S$','PVA','$\\varepsilon$']):
    ax=axes[k]
    for j in range(min(6,L1.nout)):
        phi=L1.W[j,k]*(xs/(1+np.exp(-xs)))+Bx@L1.C[j,k,:]
        ax.plot(xs,phi,color=C[j%4],lw=1.2,alpha=.9)
    ax.set_title('edge from '+lab,fontsize=9); ax.set_xlabel('standardised input',fontsize=8.5)
    if k==0: ax.set_ylabel('$\\phi_{ij}(x)$',fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=.25,lw=.5)
fig.tight_layout(); save(fig,'p2fig02_activations',D2)

# ---------------- P2 Fig 3: parity grid ----------------
names=[n for n in R.model]
sel=[n for n in ['PIKAN (data only)','DeepONet','XGBoost','MLP-PINN ($\\lambda=1,1,1$)'] if n in names]
sel+= [n for n in names if n not in sel][:4-len(sel)]
fig,axes=plt.subplots(1,4,figsize=(12.4,3.3))
for ax,nm in zip(axes,sel):
    p=z[nm]
    ax.scatter(yq,p,s=5,color=C[1],alpha=.5,edgecolors='none')
    lo,hi=0,max(yq.max(),p.max())*1.05
    ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
    r=R[R.model==nm].iloc[0]
    ax.set_title('%s\n$R^2$=%.4f, RMSE=%.2f'%(nm.replace('$\\lambda=1,1,1$','full physics'),r.R2,r.RMSE),fontsize=7.2)
    ax.set_xlabel('measured (MPa)',fontsize=8); ax.tick_params(labelsize=7.5)
axes[0].set_ylabel('predicted (MPa)',fontsize=8)
fig.tight_layout(); save(fig,'p2fig03_parity',D2)

# ---------------- P2 Fig 4: smoothness (key result) ----------------
fig,axes=plt.subplots(1,2,figsize=(10.6,3.6))
o=R.sort_values('osc_index')
ax=axes[0]
ax.barh(range(len(o)),o.osc_index,color=[C[2] if 'PIKAN' in m or 'DeepONet' in m or 'PINO' in m else C[0] for m in o.model])
ax.set_yticks(range(len(o))); ax.set_yticklabels([m.replace(' ($\\lambda=1,1,1$)','') for m in o.model],fontsize=7.5)
ax.set_xlabel('oscillation index ($\\times10^{3}$)',fontsize=9)
ax.set_title('(a) curve smoothness (lower is better)',fontsize=9); ax.tick_params(labelsize=8); ax.grid(alpha=.25,lw=.5,axis='x')
o2=R.sort_values('dmg_violations')
ax=axes[1]
ax.barh(range(len(o2)),o2.dmg_violations,color=[C[2] if 'PIKAN' in m or 'DeepONet' in m or 'PINO' in m else C[0] for m in o2.model])
ax.set_yticks(range(len(o2))); ax.set_yticklabels([m.replace(' ($\\lambda=1,1,1$)','') for m in o2.model],fontsize=7.5)
ax.set_xlabel('damage-monotonicity violations per curve',fontsize=9)
ax.set_title('(b) physical admissibility',fontsize=9); ax.tick_params(labelsize=8); ax.grid(alpha=.25,lw=.5,axis='x')
fig.tight_layout(); save(fig,'p2fig04_smoothness',D2)

# ---------------- P2 Fig 5: accuracy vs parameter count ----------------
fig,ax=plt.subplots(figsize=(6.6,4.0))
for _,r in R.iterrows():
    if not np.isfinite(r.params): continue
    mk='*' if 'PIKAN' in r.model else ('s' if 'DeepONet' in r.model or 'PINO' in r.model else 'o')
    ax.scatter(r.params,r.RMSE,s=150 if mk=='*' else 45,marker=mk,color=C[1] if 'PIKAN' in r.model else (C[2] if 'Deep' in r.model or 'PINO' in r.model else C[3]),zorder=3)
    ax.annotate(r.model.replace(' ($\\lambda=1,1,1$)',''),(r.params,r.RMSE),textcoords='offset points',
                xytext=(6,4),fontsize=7)
ax.set_xscale('log'); ax.set_xlabel('trainable parameters',fontsize=9); ax.set_ylabel('blind RMSE (MPa)',fontsize=9)
ax.grid(alpha=.25,lw=.5); ax.tick_params(labelsize=8)
fig.tight_layout(); save(fig,'p2fig05_efficiency',D2)

# ---------------- P2 Fig 6 / P3 Fig 5: multisplit ----------------
fig,ax=plt.subplots(figsize=(6.8,3.8))
order=M.groupby('model').RMSE.mean().sort_values().index.tolist()
for i,m in enumerate(order):
    v=M[M.model==m].RMSE.values
    jit=np.random.RandomState(i).uniform(-.09,.09,len(v))
    ax.scatter(np.full(len(v),i)+jit,v,s=22,color=C[i%4],alpha=.85,edgecolors='none',zorder=3)
    ax.plot([i-.26,i+.26],[np.median(v)]*2,color='#222',lw=1.8,zorder=4)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=18,ha='right',fontsize=8)
ax.set_ylabel('blind RMSE (MPa)',fontsize=9); ax.grid(alpha=.25,lw=.5); ax.tick_params(labelsize=8)
fig.tight_layout(); save(fig,'p2fig06_multisplit',D2)
import shutil; shutil.copy(os.path.join(D2,'p2fig06_multisplit.pgf'),os.path.join(D3,'p3fig05_multisplit.pgf'))

# ---------------- P2 Fig 7 / P3 Fig 3: blind curves ----------------
ks=sorted(set(zip(B_te.nS,B_te.PVA)))
fig,axes=plt.subplots(1,len(ks),figsize=(12.4,2.9),sharey=True)
for ax,k in zip(axes,ks):
    g=B_te[(B_te.nS==k[0])&(B_te.PVA==k[1])].sort_values('strain_pct')
    ax.plot(g.strain_um_m,g.stress_MPa,'k-',lw=1.6,label='experiment')
    msk=(Xq[:,0]==k[0])&(Xq[:,1]==k[1])
    ax.plot(B_te[(B_te.nS==k[0])&(B_te.PVA==k[1])].sort_values('strain_pct').strain_um_m,
            z['PIKAN (data only)'][msk][np.argsort(g.strain_um_m)],'--',color=C[1],lw=1.3,label='PIKAN')
    ax.plot(g.strain_um_m,z['DeepONet'][msk][np.argsort(g.strain_um_m)],':',color=C[2],lw=1.5,label='DeepONet')
    ax.set_title('$n_S$='+str(int(k[0]))+'\\% / PVA='+('%.1f'%k[1])+'\\%',fontsize=8)
    ax.set_xlabel('strain ($\\mu\\varepsilon$)',fontsize=8); ax.tick_params(labelsize=7.5)
axes[0].set_ylabel('stress (MPa)',fontsize=8.5); axes[0].legend(fontsize=6.5,frameon=False)
fig.tight_layout(); save(fig,'p2fig07_curves',D2)
shutil.copy(os.path.join(D2,'p2fig07_curves.pgf'),os.path.join(D3,'p3fig03_curves.pgf'))
shutil.copy(os.path.join(D2,'p2fig03_parity.pgf'),os.path.join(D3,'p3fig04_parity.pgf'))

# ---------------- P3 Fig 1: operator architecture ----------------
fig,ax=plt.subplots(figsize=(10.6,3.4)); ax.set_xlim(0,100); ax.set_ylim(0,40); ax.axis('off')
def box(x,y,w,h,t,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.0',fc=fc,ec='#54606b',lw=.9))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8.2,linespacing=1.3)
box(2,22,20,10,'branch net\n$[n_S,PVA]\\to\\mathbb{R}^{p}$','#E8F1F8')
box(2,6,20,10,'trunk net\n$[\\varepsilon]\\to\\mathbb{R}^{p}$','#EAF6F1')
box(30,14,16,10,'inner product\n$\\sum_j B_j T_j+b_0$','#FFF6E5')
box(54,14,20,10,'operator\n$\\sigma(m,\\varepsilon)$','#FDECEA')
box(80,14,17,10,'whole curve\nin one pass','#E8F1F8')
for (x0,y0,x1,y1) in [(22,27,30,20),(22,11,30,18),(46,19,54,19),(74,19,80,19)]:
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=9,color='#54606b',lw=1.0))
ax.text(12,35,'branch: one vector per mix',ha='center',fontsize=7.5,color='#333')
ax.text(12,3,'trunk: one basis over strain',ha='center',fontsize=7.5,color='#333')
fig.tight_layout(); save(fig,'p3fig01_operator',D3)

# ---------------- P3 Fig 2: learned trunk modes ----------------
mb=np.unique(Xs[:,:2],axis=0)[:6]
fig,axes=plt.subplots(1,2,figsize=(10.6,3.5))
ax=axes[0]
eps=np.linspace(0,3.0,200)
i2=2; Te=don.trunk.forward(((eps-don.scX.mean_[i2])/don.scX.scale_[i2]).reshape(-1,1))
for j in range(min(6,Te.shape[1])):
    ax.plot(eps,Te[:,j],color=C[j%4],lw=1.3)
ax.set_title('(a) learned trunk basis $T_j(\\varepsilon)$',fontsize=9)
ax.set_xlabel('Strain ($\\times10^{3}\\,\\mu\\varepsilon$)',fontsize=9); ax.set_ylabel('$T_j$',fontsize=9); ax.tick_params(labelsize=8); ax.grid(alpha=.25,lw=.5)
ax=axes[1]
mbs=(mb-don.scX.mean_[:2])/don.scX.scale_[:2]
Bc=don.branch.forward(mbs)
for j in range(min(6,Bc.shape[1])):
    ax.plot(range(len(mb)),Bc[:,j],'o-',color=C[j%4],lw=1.2,ms=3.5)
ax.set_title('(b) branch coefficients $B_j(m)$',fontsize=9)
ax.set_xlabel('withheld-mixture index',fontsize=9); ax.set_ylabel('$B_j$',fontsize=9); ax.tick_params(labelsize=8); ax.grid(alpha=.25,lw=.5)
fig.tight_layout(); save(fig,'p3fig02_modes',D3)

# ---------------- P3 Fig 6: zero-shot generation over the design space ----------------
fig,axes=plt.subplots(1,4,figsize=(12.4,2.9),sharey=True)
for ax,pva in zip(axes,[0.5,1.0,1.5,2.0]):
    for ns in [0,1,2,3,4,5]:
        eps=np.linspace(0,3.0,120)
        Xg=np.column_stack([np.full(len(eps),ns),np.full(len(eps),pva),eps])
        ax.plot(eps*1000,don.predict(Xg),color=C[[0,1,2,3,0,1][ns]],lw=1.3,
                label=('$n_S$='+str(ns)+'\\%') if pva==0.5 else None)
    ax.set_title('PVA = '+('%.1f'%pva)+'\\%',fontsize=8.5); ax.set_xlabel('strain ($\\mu\\varepsilon$)',fontsize=8)
    ax.tick_params(labelsize=7.5)
axes[0].set_ylabel('stress (MPa)',fontsize=8.5); axes[0].legend(fontsize=6.5,frameon=False,ncol=2)
fig.tight_layout(); save(fig,'p3fig06_zeroshot',D3)

# ---------------- P3 Fig 7: efficiency (shared numbers) ----------------
shutil.copy(os.path.join(D2,'p2fig05_efficiency.pgf'),os.path.join(D3,'p3fig07_efficiency.pgf'))
print('FIGURES WRITTEN to',D2,'and',D3)
print(sorted(os.listdir(D2))); print(sorted(os.listdir(D3)))
