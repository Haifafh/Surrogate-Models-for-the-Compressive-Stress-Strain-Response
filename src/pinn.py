#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics-informed neural network (numpy) for the ECC nS/PVA stress-strain surrogate.

Inputs  x = [nanosilica (%), PVA (%), axial strain (%)]      Output = stress (MPa)

Physics terms added to the data MSE (soft penalties, all evaluated in standardised units
so that the gradients stay well conditioned):
  (1) elastic-branch pin   dsigma/deps |_(eps->0)  ~=  E(nS,PVA)      [central difference]
  (2) damage consistency   sigma = (1-D) E eps ,  D in [0,1]          [continuum damage]
  (3) curvature smoothness suppresses random oscillation -> smooth hardening + softening
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

H_PCT = 0.02          # strain step (in %) used for the finite-difference stiffness residual


def _E_quad(coef, ns, pva):
    """Quadratic modulus surface E(nS,PVA) [MPa]."""
    ns = np.asarray(ns, float); pva = np.asarray(pva, float)
    return (coef[0] + coef[1] * ns + coef[2] * pva +
            coef[3] * ns ** 2 + coef[4] * pva ** 2 + coef[5] * ns * pva)


class PINN:
    def __init__(self, hidden=(64, 64, 32), lr=5e-3, epochs=15000, batch=64,
                 lam_E=1.0, lam_dmg=1.0, lam_sm=1.0, seed=0):
        self.hidden = list(hidden); self.lr = lr; self.epochs = epochs
        self.batch = batch
        self.lam_E, self.lam_dmg, self.lam_sm = lam_E, lam_dmg, lam_sm
        self.seed = seed
        self.coef = None

    # ------------------------------------------------------------------ net
    def _init_w(self, nin):
        rng = np.random.RandomState(self.seed)
        L = [nin] + self.hidden + [1]
        self.W = []
        for i in range(len(L) - 1):
            lim = np.sqrt(6.0 / (L[i] + L[i + 1]))          # Glorot uniform (tanh-friendly)
            self.W.append(rng.uniform(-lim, lim, (L[i + 1], L[i])))
        self.b = [np.zeros(L[i + 1]) for i in range(len(L) - 1)]

    def _forw(self, X):
        W, b = self.W, self.b
        a1 = np.tanh(X @ W[0].T + b[0])
        a2 = np.tanh(a1 @ W[1].T + b[1])
        a3 = np.tanh(a2 @ W[2].T + b[2])
        return a3 @ W[3].T + b[3], (a1, a2, a3)

    def _back(self, X, acts, dout):
        a1, a2, a3 = acts; W = self.W
        gz3 = (dout @ W[3]) * (1 - a3 ** 2)
        gz2 = (gz3 @ W[2]) * (1 - a2 ** 2)
        gz1 = (gz2 @ W[1]) * (1 - a1 ** 2)
        gw = [gz1.T @ X, gz2.T @ a1, gz3.T @ a2, dout.T @ a3]
        gb = [gz1.sum(0), gz2.sum(0), gz3.sum(0), dout.sum(0)]
        return gw, gb

    @staticmethod
    def _acc(gw, gb, add):
        for i in range(4):
            gw[i] = gw[i] + add[0][i]; gb[i] = gb[i] + add[1][i]
        return gw, gb

    # ------------------------------------------------------------- loss + grads
    def _grads(self, X, y, Xc, Xch, h_sc, Ec, epsf, targ):
        W = self.W
        out, acts = self._forw(X)
        dout = 2.0 * (out - y) / len(X)
        gw, gb = self._back(X, acts, dout)

        if self.lam_E == 0 and self.lam_dmg == 0 and self.lam_sm == 0:
            return gw, gb

        oc, ac = self._forw(Xc)
        oh, ah = self._forw(Xch)
        nc = len(Xc)
        epsfc = epsf.reshape(-1, 1)          # (nc,1) - keep every operand column-shaped
        Ecc = Ec.reshape(-1, 1)
        targc = targ.reshape(-1, 1)

        if self.lam_E > 0:                                   # (1) elastic stiffness pin
            m = epsfc <= 6.0e-4
            if m.any():
                dstd = (oh - oc) / h_sc                      # d sigma_std / d x_n2
                t = np.where(m, targc, 1.0)
                rE = np.where(m, (dstd - targc) / np.maximum(np.abs(t), 1e-6), 0.0)
                c = self.lam_E * 2.0 * rE / (nc * np.maximum(np.abs(t), 1e-6) * h_sc)
                gw, gb = self._acc(gw, gb, self._back(Xc, ac, -c))
                gw, gb = self._acc(gw, gb, self._back(Xch, ah, c))

        if self.lam_dmg > 0:                                 # (2) damage consistency
            m2 = epsfc >= 1.0e-3
            if m2.any():
                Ef = np.where(m2, Ecc, 1e9); ef = np.where(m2, epsfc, 1.0)
                D = 1.0 - (oc * self.sigy) / (Ef * ef)
                rD = np.where(m2, np.clip(np.clip(D - 1.0, 0, None) + np.clip(-D, 0, None), 0, 2.0), 0.0)
                dL = self.lam_dmg * 2.0 * rD / nc * (-self.sigy / (Ef * ef))
                gw, gb = self._acc(gw, gb, self._back(Xc, ac, dL))

        if self.lam_sm > 0 and nc > 2:                        # (3) curvature smoothness
            order = np.lexsort((epsf, Xc[:, 1], Xc[:, 0]))
            s = oc.ravel()[order]
            rS = np.zeros_like(s); rS[1:-1] = s[2:] - 2.0 * s[1:-1] + s[:-2]
            gS = np.zeros_like(s)
            gS[:-2] += 2.0 * rS[1:-1]; gS[1:-1] += -4.0 * rS[1:-1]; gS[2:] += 2.0 * rS[1:-1]
            dS = np.zeros_like(s); dS[order] = gS
            gw, gb = self._acc(gw, gb, self._back(Xc, ac, (self.lam_sm * dS / nc)[:, None]))
        return gw, gb

    # ------------------------------------------------------------------- train
    def fit(self, Xs, ys, E_target, phys_curves, epochs=None):
        scX = StandardScaler().fit(Xs); scY = StandardScaler().fit(ys.reshape(-1, 1))
        self.scX, self.scY = scX, scY
        self.sigy = float(scY.scale_[0]); self.epsx = float(scX.scale_[2])
        self._init_w(Xs.shape[1])
        Xn = scX.transform(Xs); yn = scY.transform(ys.reshape(-1, 1))

        rec = [(ns, pva, float(a)) for (ns, pva), tab in phys_curves for a, _ in tab]
        raw = np.array(rec)
        Xc = scX.transform(raw)
        Xch = Xc.copy(); Xch[:, 2] += H_PCT / self.epsx
        Ec = _E_quad(self.coef, raw[:, 0], raw[:, 1])
        epsf = raw[:, 2] / 100.0
        targ = (Ec * self.epsx / (self.sigy * 100.0)).reshape(-1, 1)

        if epochs: self.epochs = epochs
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(bb) for bb in self.b]; vb = [np.zeros_like(bb) for bb in self.b]
        b1, b2, ep = 0.9, 0.999, 1e-8
        lr = self.lr
        rng = np.random.RandomState(self.seed + 999)
        n = len(Xn)
        for t in range(1, self.epochs + 1):
            if t % 3000 == 0:
                lr *= 0.6                     # step decay -> stable late-stage convergence
            if self.batch and self.batch < n:
                sel = rng.choice(n, self.batch, replace=False)
                Xb, yb = Xn[sel], yn[sel]
            else:
                Xb, yb = Xn, yn
            gw, gb = self._grads(Xb, yb, Xc, Xch, H_PCT / self.epsx, Ec, epsf, targ)
            for i in range(len(self.W)):
                mW[i] = b1 * mW[i] + (1 - b1) * gw[i]; vW[i] = b2 * vW[i] + (1 - b2) * gw[i] ** 2
                mb[i] = b1 * mb[i] + (1 - b1) * gb[i]; vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
                self.W[i] -= lr * (mW[i] / (1 - b1 ** t)) / (np.sqrt(vW[i] / (1 - b2 ** t)) + ep)
                self.b[i] -= lr * (mb[i] / (1 - b1 ** t)) / (np.sqrt(vb[i] / (1 - b2 ** t)) + ep)
        return self

    def predict(self, Xs, scX=None, scY=None):
        scX = self.scX if scX is None else scX
        scY = self.scY if scY is None else scY
        y, _ = self._forw(scX.transform(np.asarray(Xs, float)))
        return scY.inverse_transform(y).ravel()
