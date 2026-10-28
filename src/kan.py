#!/usr/bin/env python3
"""Kolmogorov-Arnold Network (B-spline) with optional physics residuals (PIKAN).

Layer:  y_j = sum_i phi_{ij}(x_i),   phi_{ij}(z) = W[j,i]*silu(z) + sum_k C[j,i,k]*B_k(z)
B_k are B-spline basis functions of order p on a uniform grid (local support).
"""
import numpy as np

def knot_vector(lo, hi, G, p):
    return np.concatenate([[lo] * (p + 1), np.linspace(lo, hi, G + 1)[1:-1], [hi] * (p + 1)])

def bspline(x, t, p, deriv=0):
    """B-spline basis (deriv=0) or its first derivative (deriv=1), shape (n, nb)."""
    x = np.asarray(x, float).ravel(); n = x.size; m = t.size
    B = np.zeros((n, m - 1))
    for i in range(m - 1):
        B[:, i] = ((x >= t[i]) & (x < t[i + 1])).astype(float)
    B[x >= t[-1], -1] = 1.0
    top = p if deriv == 0 else p - 1
    for k in range(1, top + 1):
        nb = m - k - 1
        Bn = np.zeros((n, nb))
        for i in range(nb):
            d1 = t[i + k] - t[i]; d2 = t[i + k + 1] - t[i + 1]
            a = (x - t[i]) / d1 if d1 > 0 else np.zeros(n)
            b = (t[i + k + 1] - x) / d2 if d2 > 0 else np.zeros(n)
            Bn[:, i] = a * B[:, i] + b * B[:, i + 1]
        B = Bn
    if deriv == 0:
        return B
    nb = m - p - 1
    dB = np.zeros((n, nb))
    for i in range(nb):
        d1 = t[i + p] - t[i]; d2 = t[i + p + 1] - t[i + 1]
        a = p / d1 if d1 > 0 else 0.0
        b = p / d2 if d2 > 0 else 0.0
        dB[:, i] = a * B[:, i] - b * B[:, i + 1]
    return dB

def _silu(x):  return x / (1.0 + np.exp(-np.clip(x, -60, 60)))
def _dsilu(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
    return s + x * s * (1 - s)

class KANLayer:
    def __init__(self, nin, nout, G=8, p=3, lo=-3.0, hi=3.0, seed=0):
        self.nin, self.nout, self.G, self.p = nin, nout, G, p
        self.t = knot_vector(lo, hi, G, p)
        self.nb = len(self.t) - p - 1
        rng = np.random.RandomState(seed)
        self.W = rng.normal(0, 0.1, (nout, nin))
        self.C = rng.normal(0, 0.1, (nout, nin, self.nb))

    def forward(self, X, cache=None):
        n = X.shape[0]
        Y = np.zeros((n, self.nout))
        Bs, dBs, S, dS = [], [], [], []
        for i in range(self.nin):
            xi = np.clip(X[:, i], self.t[0], self.t[-1])
            B = bspline(xi, self.t, self.p, 0); dB = bspline(xi, self.t, self.p, 1)
            Bs.append(B); dBs.append(dB)
            S.append(_silu(xi)); dS.append(_dsilu(xi))
            Y += S[-1][:, None] * self.W[:, i][None, :] + B @ self.C[:, i, :].T
        if cache is not None:
            cache.update(Bs=Bs, dBs=dBs, S=S, dS=dS, X=X, n=n)
        return Y

    def backward(self, dY, cache):
        dW = np.zeros_like(self.W); dC = np.zeros_like(self.C)
        dX = np.zeros_like(cache['X'])
        for i in range(self.nin):
            dW[:, i] = (dY * cache['S'][i][:, None]).sum(0)
            dC[:, i, :] = dY.T @ cache['Bs'][i]
            term1 = cache['dS'][i] * (dY @ self.W[:, i])
            term2 = ((dY @ self.C[:, i, :]) * cache['dBs'][i]).sum(1)
            dX[:, i] = term1 + term2
        return dW, dC, dX

class KAN:
    def __init__(self, dims=(3, 12, 1), G=8, p=3, lo=-3.0, hi=3.0, seed=0,
                 lr=5e-3, epochs=6000, batch=64, lam_E=0.0, lam_dmg=0.0, lam_sm=0.0):
        self.layers = [KANLayer(dims[i], dims[i + 1], G, p, lo, hi, seed + i)
                       for i in range(len(dims) - 1)]
        self.lr, self.epochs, self.batch = lr, epochs, batch
        self.lam_E, self.lam_dmg, self.lam_sm = lam_E, lam_dmg, lam_sm
        self.seed = seed; self.coef = None

    def params(self):
        return [(L.W, L.C) for L in self.layers]

    def forward(self, X):
        caches = []; h = X
        for L in self.layers:
            c = {}; h = L.forward(h, c); caches.append(c)
        return h, caches

    def predict(self, X):
        Xs = self.scX.transform(np.asarray(X, float))
        y, _ = self.forward(Xs)
        return self.scY.inverse_transform(y).ravel()

    def _grads(self, Xb, yb, Xc=None, Xch=None, h_sc=0.0, Ec=None, epsf=None, targ=None):
        """Returns (dW,dC) per layer for the full loss (data + physics)."""
        n = Xb.shape[0]
        out, caches = self.forward(Xb)
        dout = 2.0 * (out - yb) / n
        grads = [None] * len(self.layers)
        d = dout
        for li in range(len(self.layers) - 1, -1, -1):
            dW, dC, d = self.layers[li].backward(d, caches[li])
            grads[li] = (dW, dC)
        if Xc is None or (self.lam_E == 0 and self.lam_dmg == 0 and self.lam_sm == 0):
            return grads
        # ---- physics: central difference for the stiffness pin (as in the MLP-PINN)
        nc = Xc.shape[0]
        sigy = float(self.scY.scale_[0])
        if self.lam_E > 0:
            oc, _ = self.forward(Xc); oh, _ = self.forward(Xch)
            dstd = (oh - oc) / h_sc
            m = (epsf <= 6.0e-4).reshape(-1, 1)
            t = np.where(m, targ, 1.0)
            rE = np.where(m, (dstd - targ) / np.maximum(np.abs(t), 1e-6), 0.0)
            c = self.lam_E * 2.0 * rE / (nc * np.maximum(np.abs(t), 1e-6) * h_sc)
            self._acc(grads, Xc, -c, caches); self._acc(grads, Xch, c, None)
        if self.lam_dmg > 0:
            oc, _ = self.forward(Xc)
            m2 = (epsf >= 1.0e-3).reshape(-1, 1)
            Ef = np.where(m2, Ec.reshape(-1, 1), 1e9); ef = np.where(m2, epsf.reshape(-1, 1), 1.0)
            D = 1.0 - (oc * sigy) / (Ef * ef)
            rD = np.where(m2, np.clip(D - 1.0, 0, None) + np.clip(-D, 0, None), 0.0)
            dL = self.lam_dmg * 2.0 * rD / nc * (-sigy / (Ef * ef))
            self._acc(grads, Xc, dL, caches)
        if self.lam_sm > 0:
            oc, _ = self.forward(Xc)
            order = np.lexsort((epsf.ravel(), Xc[:, 1], Xc[:, 0]))
            s = oc.ravel()[order]
            rS = np.zeros_like(s); rS[1:-1] = s[2:] - 2 * s[1:-1] + s[:-2]
            gS = np.zeros_like(s)
            gS[:-2] += 2 * rS[1:-1]; gS[1:-1] += -4 * rS[1:-1]; gS[2:] += 2 * rS[1:-1]
            dS = np.zeros_like(s); dS[order] = gS
            self._acc(grads, Xc, (self.lam_sm * dS / nc)[:, None], caches)
        return grads

    def _acc(self, grads, X, seed_dy, caches):
        """Accumulate a physics gradient contribution flowing from output seed_dy at input X."""
        out, caches2 = self.forward(X)
        d = seed_dy
        for li in range(len(self.layers) - 1, -1, -1):
            dW, dC, d = self.layers[li].backward(d, caches2[li])
            grads[li] = (grads[li][0] + dW, grads[li][1] + dC)

    def fit(self, Xs, ys, E_target=None, phys_curves=None, H_PCT=0.02, epochs=None):
        from sklearn.preprocessing import StandardScaler
        self.scX = StandardScaler().fit(Xs); self.scY = StandardScaler().fit(ys.reshape(-1, 1))
        Xn = self.scX.transform(Xs); yn = self.scY.transform(ys.reshape(-1, 1))
        epsx = float(self.scX.scale_[2]); sigy = float(self.scY.scale_[0])
        Xc = Xch = None; Ec = epsf = targ = None; h_sc = 0.0
        if phys_curves is not None and (self.lam_E or self.lam_dmg or self.lam_sm):
            rec = np.array([(ns, pva, float(a)) for (ns, pva), tab in phys_curves for a, _ in tab])
            Xc = self.scX.transform(rec)
            Xch = Xc.copy(); Xch[:, 2] += H_PCT / epsx
            Ec = np.array([self._E(rec[k, 0], rec[k, 1]) for k in range(len(rec))])
            epsf = rec[:, 2] / 100.0
            targ = (Ec * epsx / (sigy * 100.0)).reshape(-1, 1)
            h_sc = H_PCT / epsx
        m = [(np.zeros_like(L.W), np.zeros_like(L.C)) for L in self.layers]
        v = [(np.zeros_like(L.W), np.zeros_like(L.C)) for L in self.layers]
        b1, b2, ep = 0.9, 0.999, 1e-8
        lr = self.lr; rng = np.random.RandomState(self.seed + 999)
        n = len(Xn); E = epochs or self.epochs
        for t in range(1, E + 1):
            if t % 3000 == 0: lr *= 0.6
            if self.batch and self.batch < n:
                sel = rng.choice(n, self.batch, replace=False); Xb, yb = Xn[sel], yn[sel]
            else:
                Xb, yb = Xn, yn
            grads = self._grads(Xb, yb, Xc, Xch, h_sc, Ec, epsf, targ)
            for li, (L, (gW, gC)) in enumerate(zip(self.layers, grads)):
                for i, (g, mm, vv, P) in enumerate([(gW, m[li][0], v[li][0], L.W),
                                                     (gC, m[li][1], v[li][1], L.C)]):
                    mm[...] = b1 * mm + (1 - b1) * g
                    vv[...] = b2 * vv + (1 - b2) * g ** 2
                    P[...] -= lr * (mm / (1 - b1 ** t)) / (np.sqrt(vv / (1 - b2 ** t)) + ep)
        return self

    def _E(self, ns, pva):
        c = self.coef
        return (c[0] + c[1] * ns + c[2] * pva + c[3] * ns ** 2 + c[4] * pva ** 2 + c[5] * ns * pva)

    def n_params(self):
        return sum(L.W.size + L.C.size for L in self.layers)
