#!/usr/bin/env python3
"""DeepONet / PINO operator: branch net (mix design) x trunk net (strain) -> stress curve.

sigma(m, eps) = sum_j B_j(m) T_j(eps) + b0
Trained on (nS,PVA,strain)->stress but factorised, so a whole curve is produced in one
shot for a new mix. PINO variant adds physics residuals on the output curve.
"""
import numpy as np

def _silu(x): return x/(1.0+np.exp(-np.clip(x,-60,60)))

class MLP:
    def __init__(self, dims, act='tanh', seed=0):
        rng = np.random.RandomState(seed)
        self.W = []; self.b = []
        for i in range(len(dims)-1):
            lim = np.sqrt(6.0/(dims[i]+dims[i+1]))
            self.W.append(rng.uniform(-lim, lim, (dims[i+1], dims[i])))
            self.b.append(np.zeros(dims[i+1]))
        self.act = act
    def _a(self, z): return np.tanh(z) if self.act=='tanh' else _silu(z)
    def _da(self, z, a): return (1-a**2) if self.act=='tanh' else None
    def forward(self, X, cache=None):
        h = X; acts=[h]
        for i,(W,b) in enumerate(zip(self.W,self.b)):
            z = h@W.T+b; h = self._a(z); acts.append(h)
        if cache is not None: cache['acts']=acts
        return h
    def backward(self, dout, cache):
        acts = cache['acts']; dW=[]; db=[]; d = dout
        for i in range(len(self.W)-1, -1, -1):
            a = acts[i+1]; z = acts[i]@self.W[i].T+self.b[i]
            if self.act=='tanh': g = d*(1-a**2)
            else:
                s = 1/(1+np.exp(-np.clip(z,-60,60))); g = d*(s+z*s*(1-s))
            dW.insert(0, g.T@acts[i]); db.insert(0, g.sum(0)); d = g@self.W[i]
        return dW, db, d

class DeepONet:
    def __init__(self, p=24, hb=(32,32), ht=(32,32), seed=0, lr=5e-3, epochs=6000,
                 batch=128, lam_E=0.0, lam_dmg=0.0):
        self.branch = MLP((2,)+hb+(p,), 'tanh', seed)
        self.trunk  = MLP((1,)+ht+(p,), 'tanh', seed+101)
        self.b0 = 0.0; self.p = p
        self.lr, self.epochs, self.batch = lr, epochs, batch
        self.lam_E, self.lam_dmg = lam_E, lam_dmg
        self.seed = seed; self.coef=None
    def _split(self, X):
        return X[:, :2], X[:, 2:3]
    def forward(self, X, cache=None):
        mb, ms = self._split(X)
        cb={}; ct={}
        B = self.branch.forward(mb, cb); T = self.trunk.forward(ms, ct)
        if cache is not None: cache.update(cb=cb, ct=ct, B=B, T=T)
        return (B*T).sum(1, keepdims=True) + self.b0
    def predict(self, X):
        Xs = self.scX.transform(np.asarray(X,float))
        y = self.forward(Xs)
        return self.scY.inverse_transform(y).ravel()
    def n_params(self):
        f=lambda m: sum(w.size for w in m.W)+sum(b.size for b in m.b)
        return f(self.branch)+f(self.trunk)
    def _acc(self, gb, gt, dY, X):
        c={}; self.forward(X, c)
        dB = dY*c['T']; dT = dY*c['B']
        dWb, dbb, _ = self.branch.backward(dB, c['cb'])
        dWt, dbt, _ = self.trunk.backward(dT, c['ct'])
        for i in range(len(dWb)): gb[0][i]+=dWb[i]; gb[1][i]+=dbb[i]
        for i in range(len(dWt)): gt[0][i]+=dWt[i]; gt[1][i]+=dbt[i]
    def fit(self, Xs, ys, H_PCT=0.02, epochs=None):
        from sklearn.preprocessing import StandardScaler
        self.scX = StandardScaler().fit(Xs); self.scY = StandardScaler().fit(ys.reshape(-1,1))
        Xn = self.scX.transform(Xs); yn = self.scY.transform(ys.reshape(-1,1))
        epsx=float(self.scX.scale_[2]); sigy=float(self.scY.scale_[0])
        # collocation for the elastic pin: one point per mixture at small strain
        mixes = np.unique(Xs[:,:2], axis=0)
        Xc = self.scX.transform(np.column_stack([mixes, np.full(len(mixes), 0.05)]))
        Xch = Xc.copy(); Xch[:,2] += H_PCT/epsx
        Ec = np.array([self._E(m[0],m[1]) for m in mixes])
        targ = (Ec*epsx/(sigy*100.0)).reshape(-1,1)
        mb=[(np.zeros_like(w),np.zeros_like(b)) for w,b in zip(self.branch.W,self.branch.b)]
        vb=[(np.zeros_like(w),np.zeros_like(b)) for w,b in zip(self.branch.W,self.branch.b)]
        mt=[(np.zeros_like(w),np.zeros_like(b)) for w,b in zip(self.trunk.W,self.trunk.b)]
        vt=[(np.zeros_like(w),np.zeros_like(b)) for w,b in zip(self.trunk.W,self.trunk.b)]
        b1,b2,ep=0.9,0.999,1e-8; lr=self.lr
        rng=np.random.RandomState(self.seed+7); n=len(Xn); E=epochs or self.epochs
        h_sc=H_PCT/epsx
        for t in range(1,E+1):
            if t%3000==0: lr*=0.6
            if self.batch and self.batch<n:
                sel=rng.choice(n,self.batch,replace=False); Xb,yb=Xn[sel],yn[sel]
            else: Xb,yb=Xn,yn
            c={}; out=self.forward(Xb,c)
            dY = 2.0*(out-yb)/len(Xb)
            dB = dY*c['T']; dT = dY*c['B']
            dWb,dbb,_ = self.branch.backward(dB,c['cb'])
            dWt,dbt,_ = self.trunk.backward(dT,c['ct'])
            gb=(dWb,dbb); gt=(dWt,dbt)
            if self.lam_E>0:
                oc=self.forward(Xc); oh=self.forward(Xch)
                dstd=(oh-oc)/h_sc
                rE=(dstd-targ)/np.maximum(np.abs(targ),1e-6)
                cc=self.lam_E*2.0*rE/(len(Xc)*np.maximum(np.abs(targ),1e-6)*h_sc)
                self._acc(gb,gt,-cc,Xc); self._acc(gb,gt,cc,Xch)
            for grp,ms_,vs_,mod in [(gb,mb,vb,self.branch),(gt,mt,vt,self.trunk)]:
                for i in range(len(mod.W)):
                    for arr,g,mm,vv in [(mod.W[i],grp[0][i],ms_[i][0],vs_[i][0]),
                                        (mod.b[i],grp[1][i],ms_[i][1],vs_[i][1])]:
                        mm[...]=b1*mm+(1-b1)*g; vv[...]=b2*vv+(1-b2)*g**2
                        arr[...]-=lr*(mm/(1-b1**t))/(np.sqrt(vv/(1-b2**t))+ep)
            self.b0 -= lr*float(dY.mean())
        return self
    def _E(self,ns,pva):
        c=self.coef
        return (c[0]+c[1]*ns+c[2]*pva+c[3]*ns**2+c[4]*pva**2+c[5]*ns*pva)
