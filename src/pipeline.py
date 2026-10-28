#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECC (nanosilica + PVA) stress-strain surrogate benchmark.
Data : unzipped/curves (1).xlsx  (sheets a-f = nS 0,1,2,3,4,5 %; cols N1-N4 = PVA 0.5,1.0,1.5,2.0 %)
Units: strain axis is per-mille (value 2.4 -> 2400 um/m -> 0.24 %); stress in MPa.
Outputs: out/*.csv, out/*.json, figures/*.png|pdf
"""
import os, json, warnings, itertools
import numpy as np, pandas as pd
import openpyxl
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

RNG = np.random.RandomState(42)
np.random.seed(42)

ROOT = os.path.expanduser('~/workspace/paper')
UNZ  = os.path.join(ROOT, 'unzipped')
OUT  = os.path.join(ROOT, 'out');      os.makedirs(OUT, exist_ok=True)
FIG  = os.path.join(ROOT, 'figures');  os.makedirs(FIG, exist_ok=True)

SHEET_NS = {'a': 0.0, 'b': 1.0, 'c': 2.0, 'd': 3.0, 'e': 4.0, 'f': 5.0}
PVA_LEVELS = [0.5, 1.0, 1.5, 2.0]
TEST_COMBOS = [(1.0, 0.5), (2.0, 1.5), (3.0, 2.0), (4.0, 1.0), (5.0, 1.5)]  # blind mixtures

# ----------------------------------------------------------------------------- 1. DATA
def load_raw_curves():
    wb = openpyxl.load_workbook(os.path.join(UNZ, 'curves (1).xlsx'), data_only=True)
    curves = {}
    for sh, ns in SHEET_NS.items():
        rows = list(wb[sh].iter_rows(values_only=True))
        for j, pva in enumerate(PVA_LEVELS):
            c0, c1 = 2 * j, 2 * j + 1
            pts = []
            for r in rows[1:]:
                if len(r) <= c1: continue
                e, s = r[c0], r[c1]
                if e is None or s is None: continue
                try: e, s = float(e), float(s)
                except (TypeError, ValueError): continue
                pts.append((e, s))
            assert pts, (sh, pva)
            curves[(ns, pva)] = np.array(sorted(pts))
    return curves


def build_dataset(step=0.1):
    """Resample every experimental curve onto a common strain grid (linear interp)."""
    curves = load_raw_curves()
    recs, props = [], []
    for (ns, pva), pts in curves.items():
        ax, sig = pts[:, 0], pts[:, 1]
        grid = np.round(np.arange(0.0, ax.max() + 1e-9, step), 6)
        sg = np.interp(grid, ax, sig)
        # initial modulus: least squares over the pre-40%-peak branch
        ipk = int(np.argmax(sig)); speak = sig[ipk]; axpk = ax[ipk]
        m0 = sig <= 0.4 * speak
        if m0.sum() < 2: m0 = np.arange(len(sig)) < 3
        E0 = float(np.sum((ax[m0] * 1e-3) * sig[m0]) / np.sum((ax[m0] * 1e-3) ** 2))  # MPa
        props.append(dict(nS=ns, PVA=pva, sigma_peak=float(speak), strain_peak_um_m=float(axpk * 1e3),
                          strain_peak_pct=float(axpk / 10.0), E0_GPa=E0 / 1000.0,
                          n_points=len(grid), strain_max_um_m=float(ax.max() * 1e3)))
        for a, s in zip(grid, sg):
            recs.append(dict(nS=ns, PVA=pva, strain_um_m=a * 1e3, strain_pct=a / 10.0,
                             eps_frac=a * 1e-3, stress_MPa=float(s)))
    df = pd.DataFrame(recs).sort_values(['nS', 'PVA', 'strain_um_m']).reset_index(drop=True)
    pr = pd.DataFrame(props).sort_values(['nS', 'PVA']).reset_index(drop=True)
    return df, pr, curves


# ----------------------------------------------------------------------------- 2. PHYSICS
def fit_modulus_surface(pr, subset=None):
    """Quadratic surface E0(nS,PVA) -> used as the elastic-modulus physics target."""
    d = pr if subset is None else pr[pr.apply(lambda r: (r.nS, r.PVA) in subset, axis=1)]
    A = np.column_stack([np.ones(len(d)), d.nS, d.PVA, d.nS ** 2, d.PVA ** 2, d.nS * d.PVA])
    coef, *_ = np.linalg.lstsq(A, d.E0_GPa.values * 1000.0, rcond=None)
    return coef

def E_surface(coef, ns, pva):
    return (coef[0] + coef[1]*ns + coef[2]*pva + coef[3]*ns**2 + coef[4]*pva**2 + coef[5]*ns*pva)


def carreira_chu(speak, eps_pk, E0, npts=160, frac=1.9):
    """Physics-only (no ML) reference constitutive model: Carreira & Chu (1985)."""
    eps = np.linspace(0, frac * eps_pk, npts)
    Esec = speak / eps_pk
    E0 = max(E0, 1.05 * Esec)
    r = E0 / (E0 - Esec)
    x = eps / eps_pk
    with np.errstate(divide='ignore', invalid='ignore'):
        sig = speak * x * r / (r - 1 + x ** r)
    return eps, sig


# ----------------------------------------------------------------------------- 3. NUMPY PINN
from pinn import PINN  # noqa: E402  clean, verified implementation (see pinn.py)


class _PINN_legacy:
    """Physics-informed MLP (numpy) with exact d(sigma)/d(eps) via analytic chain rule.

    Physics loss terms
      Lin  : initial modulus pinned to E(nS,PVA)  (elastic branch)
      Ldmg : continuum-damage consistency  sigma = (1-D) E eps ,  D in [0,1], dD/deps >= 0
      Lsm  : curvature smoothness (suppresses random oscillation, enforces single peak)
    """
    def __init__(self, nin=3, hidden=(64, 64, 32), lr=3e-3, epochs=6000,
                 lam_E=0.30, lam_dmg=0.02, lam_sm=0.01, lam_out=0.0, seed=0):
        self.layers = [nin] + list(hidden) + [1]
        self.lr, self.epochs, self.seed = lr, epochs, seed
        self.lam_E, self.lam_dmg, self.lam_sm, self.lam_out = lam_E, lam_dmg, lam_sm, lam_out
        self._init()

    def _init(self):
        rng = np.random.RandomState(self.seed)
        self.W, self.b = [], []
        for i in range(len(self.layers) - 1):
            fan = self.layers[i]
            self.W.append(rng.normal(0, np.sqrt(2.0 / fan), (self.layers[i + 1], self.layers[i])))
            self.b.append(np.zeros((self.layers[i + 1],)))
        self.W = np.array(self.W, dtype=object); self.b = np.array(self.b, dtype=object)

    def _fwd(self, X):
        a, g = X, None
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W.T + b
            dz = np.ones_like(z) if i == len(self.W) - 1 else 1.0 - np.tanh(z) ** 2
            a = z if i == len(self.W) - 1 else np.tanh(z)
            g = W[:, 2][None, :] * np.ones((len(X), 1))[:, [0]] if g is None else None
            g = (g @ W.T) if i > 0 else np.repeat(W[:, 2][None, :], len(X), axis=0) * dz
            g = g * dz if i > 0 else g
        return a, g

    # --- cleaner explicit implementation (used everywhere) -------------------
    def forward(self, X):
        """returns sigma (std units) and d sigma / d x2 (std units)."""
        a, g = X, np.zeros_like(X); g[:, 2] = 1.0
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W.T + b
            if i == len(self.W) - 1:
                a = z
            else:
                a = np.tanh(z)
                g = (1.0 - a ** 2) * (g @ W.T)
        return a, g

    def predict(self, Xs, scX, scY):
        Xn = scX.transform(Xs)
        y, _ = self.forward(Xn)
        return scY.inverse_transform(y).ravel()

    def _adam(self, X, y, phys, verbose=False):
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, ep = 0.9, 0.999, 1e-8
        Xc, E_t, ns_c, pva_c, eps_c, sig_c = phys
        for t in range(1, self.epochs + 1):
            # ---- data branch
            a1 = np.tanh(X @ self.W[0].T + self.b[0])
            a2 = np.tanh(a1 @ self.W[1].T + self.b[1])
            a3 = np.tanh(a2 @ self.W[2].T + self.b[2])
            out = (a3 @ self.W[3].T + self.b[3])
            dL_dout = 2.0 * (out - y) / len(X)
            # ---- physics branch
            pc, gp = self.forward(Xc)
            dSig = gp[:, 2] * self.sigy / self.epsx          # d sigma / d eps (MPa per frac)
            epsf = eps_c
            # (a) initial stiffness pin
            mL = epsf <= 0.45e-3
            rE = np.where(mL, (dSig - E_t) / self.Eref, 0.0)
            # (b) damage consistency
            Dp = 1.0 - pc * self.sigy / (E_t * np.maximum(epsf, 1e-9))
            rD = np.clip(Dp, 0, None) * (Dp > 1.0) + np.clip(-Dp, 0, None) * (Dp < 0.0)
            # (c) smoothness (2nd difference along eps)
            order = np.lexsort((epsf, pva_c, ns_c))
            s = pc.ravel()[order]
            rS = np.zeros_like(s)
            if len(s) > 2:
                rS[1:-1] = (s[2:] - 2 * s[1:-1] + s[:-2]) / self.sigy
            dL_dpc = np.zeros_like(pc)
            dL_dpc[:, 0] = 2.0 * self.lam_E * rE / (len(pc) * self.Eref)
            dL_dpc[:, 0] += self.lam_dmg * 2.0 * rD / (len(pc) * 1.0) * (-self.sigy / (E_t * np.maximum(epsf, 1e-9)))
            tmp = np.zeros_like(s); tmp[1:-1] = 2.0 * rS[1:-1] / self.sigy
            dS = np.zeros_like(s); dS[order] = tmp
            dL_dpc[:, 0] += self.lam_sm * dS / len(pc)
            dL_dpc *= self.lam_E if False else 1.0
            # backprop physics (sigma branch only)
            dL_dout_p = dL_dpc
            g1, g2, g3 = dL_dout_p @ self.W[3], None, None
            g3 = g1 * (1 - a3 ** 2); g2 = (g3 @ self.W[2]) * (1 - a2 ** 2); g1n = (g2 @ self.W[1]) * (1 - a1 ** 2)
            gw_phys = [g1n.T @ Xc, g2.T @ a1, g3.T @ a2, dL_dout_p.T @ a3]
            gb_phys = [g1n.sum(0), g2.sum(0), g3.sum(0), dL_dout_p.sum(0)]
            # ---- data grads
            gd3 = dL_dout * (1 - a3 ** 2); gd2 = (gd3 @ self.W[3]) * (1 - a2 ** 2)
            gd1 = (gd2 @ self.W[2]) * (1 - a1 ** 2)
            gw = [gd1.T @ X, gd2.T @ a1, gd3.T @ a2, dL_dout.T @ a3]
            gb = [gd1.sum(0), gd2.sum(0), gd3.sum(0), dL_dout.sum(0)]
            for i in range(4):
                gw[i] += gw_phys[i]; gb[i] += gb_phys[i]
            # ---- adam
            for i in range(4):
                mW[i] = b1 * mW[i] + (1 - b1) * gw[i]; vW[i] = b2 * vW[i] + (1 - b2) * gw[i] ** 2
                mb[i] = b1 * mb[i] + (1 - b1) * gb[i]; vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
                mWh = mW[i] / (1 - b1 ** t); vWh = vW[i] / (1 - b2 ** t)
                mBh = mb[i] / (1 - b1 ** t); vBh = vb[i] / (1 - b2 ** t)
                self.W[i] -= self.lr * mWh / (np.sqrt(vWh) + ep)
                self.b[i] -= self.lr * mBh / (np.sqrt(vBh) + ep)

    def fit(self, Xs, ys, E_target, phys_curves, epochs=None):
        scX = StandardScaler().fit(Xs); scY = StandardScaler().fit(ys.reshape(-1, 1))
        self.scX, self.scY = scX, scY
        self.sigy = float(scY.scale_[0]); self.epsx = 1.0
        Xn = scX.transform(Xs); yn = scY.transform(ys.reshape(-1, 1))
        self.Eref = float(np.mean(E_target)) ** 2
        # collocation set: full curves of the training mixtures
        recs = []
        for (ns, pva), arr in phys_curves:
            for a, s in arr:
                recs.append((ns, pva, a * 1e-3, s))
        R = pd.DataFrame(recs, columns=['nS', 'PVA', 'eps', 'sig'])
        Xc = scX.transform(R[['nS', 'PVA', 'eps']].values if False else
                           np.column_stack([R.nS, R.PVA, R.eps * 1e3]))
        Ep = E_surface(self.coef, R.nS.values, R.PVA.values)
        if epochs: self.epochs = epochs
        self._adam(Xn, yn, (Xc, Ep, R.nS.values, R.PVA.values, R.eps.values, R.sig.values))
        return self


def train_pinn(df_tr, pr_all, coef, hidden=(64, 64, 32), epochs=15000, seed=0, batch=64,
               lr=5e-3, lam_E=1.0, lam_dmg=1.0, lam_sm=1.0):
    m = PINN(hidden=hidden, epochs=epochs, seed=seed, batch=batch, lr=lr,
             lam_E=lam_E, lam_dmg=lam_dmg, lam_sm=lam_sm)
    m.coef = coef
    Xs = df_tr[['nS', 'PVA', 'strain_pct']].values.astype(float)
    ys = df_tr['stress_MPa'].values.astype(float)
    pc = [(k, df_tr[(df_tr.nS == k[0]) & (df_tr.PVA == k[1])][['strain_pct', 'stress_MPa']].values) for k in
          df_tr.groupby(['nS', 'PVA']).groups.keys()]
    m.fit(Xs, ys, E_surface(coef, Xs[:, 0], Xs[:, 1]), pc)
    return m


PINN_TRAIN_GRID = [dict(lr=5e-3, batch=64, epochs=12000),
                   dict(lr=1e-2, batch=64, epochs=12000),
                   dict(lr=5e-3, batch=0, epochs=12000)]
PINN_GRID = [dict(lam_E=e, lam_dmg=d, lam_sm=s)
             for e in (0.0, 3.0) for d in (0.0, 0.5) for s in (0.0, 1.0)]


def calibrate_pinn(df_tr, pr_all, coef, val_frac=0.2, seed=7, verbose=True):
    """Two-stage selection on an INNER validation split of the training mixtures:
    (1) optimiser settings with the physics switched off, (2) physics weights.
    The blind test set is never touched here."""
    combos = sorted(set(zip(df_tr.nS, df_tr.PVA)))
    rng = np.random.RandomState(seed)
    nval = max(3, int(round(val_frac * len(combos))))
    val = set(combos[i] for i in rng.choice(len(combos), nval, replace=False))
    istest = df_tr.apply(lambda r: (r.nS, r.PVA) in val, axis=1)
    tr2, va = df_tr[~istest], df_tr[istest]
    Xv, yv = va[['nS', 'PVA', 'strain_pct']].values, va.stress_MPa.values
    import time

    def trial(kw):
        t0 = time.time()
        m = train_pinn(tr2, pr_all, coef, seed=0, **kw)
        e = float(np.sqrt(mean_squared_error(yv, m.predict(Xv, m.scX, m.scY))))
        if verbose:
            print(f'      calib {kw} -> val RMSE {e:.3f} MPa [{time.time()-t0:.1f}s]', flush=True)
        return e

    best_tr, best_e = None, np.inf
    for kw in PINN_TRAIN_GRID:
        e = trial(dict(kw, lam_E=0.0, lam_dmg=0.0, lam_sm=0.0))
        if e < best_e:
            best_e, best_tr = e, kw
    best_lam, best_l = None, np.inf
    for lam in PINN_GRID:
        e = trial(dict(best_tr, **lam))
        if e < best_l:
            best_l, best_lam = e, lam
    if verbose:
        print(f'    SELECTED {dict(best_tr, **best_lam)}  (inner-val RMSE {best_l:.3f} MPa)')
    return dict(best_tr, **best_lam)


# ----------------------------------------------------------------------------- 4. MODELS
def model_zoo():
    Z = {}
    Z['Linear Regression'] = make_pipeline(StandardScaler(), LinearRegression())
    Z['Polynomial Reg. (deg 3)'] = make_pipeline(StandardScaler(), PolynomialFeatures(3), LinearRegression())
    Z['Ridge Regression'] = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    Z['SVR (RBF)'] = make_pipeline(StandardScaler(), GridSearchCV(
        SVR(kernel='rbf'), {'C': [1, 10, 100, 1000], 'gamma': ['scale', 0.1, 0.5, 1.0]}, cv=3, n_jobs=-1))
    Z['k-NN (k=5)'] = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights='distance'))
    Z['Decision Tree'] = DecisionTreeRegressor(max_depth=8, random_state=42)
    Z['Random Forest'] = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    Z['Extra Trees'] = ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    Z['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    Z['HistGradient Boosting'] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=42)
    Z['Gaussian Process'] = make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=C(1.0) * RBF([1.0, 1.0, 1.0]) + WhiteKernel(1e-3), normalize_y=True, random_state=42))
    Z['ANN (shallow MLP)'] = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(32,), activation='tanh', solver='adam', max_iter=6000,
        learning_rate_init=3e-3, random_state=42))
    Z['Deep ANN (MLP 64-64-32)'] = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(64, 64, 32), activation='tanh', solver='adam', max_iter=6000,
        learning_rate_init=2e-3, random_state=42))
    try:
        from xgboost import XGBRegressor
        Z['XGBoost'] = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                                    subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=0)
    except Exception:
        pass
    return Z


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    r2 = r2_score(y, yhat)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mae = float(mean_absolute_error(y, yhat))
    m = y > 1.0
    mape = float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100) if m.any() else np.nan
    return dict(R2=float(r2), RMSE=rmse, MAE=mae, MAPE=mape)


def curve_metrics(df_te, yhat):
    d = df_te.copy(); d['pred'] = yhat
    rows = []
    for (ns, pva), g in d.groupby(['nS', 'PVA']):
        rows.append(dict(nS=ns, PVA=pva, sigma_peak_true=g.stress_MPa.max(), sigma_peak_pred=g.pred.max(),
                         eps_peak_true_um_m=g.loc[g.stress_MPa.idxmax(), 'strain_um_m'],
                         eps_peak_pred_um_m=g.loc[g.pred.idxmax(), 'strain_um_m'],
                         R2_curve=float(r2_score(g.stress_MPa, g.pred)),
                         RMSE_curve=float(np.sqrt(mean_squared_error(g.stress_MPa, g.pred)))))
    c = pd.DataFrame(rows)
    c['err_peak_pct'] = 100 * (c.sigma_peak_pred - c.sigma_peak_true) / c.sigma_peak_true
    c['err_eps_peak_pct'] = 100 * (c.eps_peak_pred_um_m - c.eps_peak_true_um_m) / c.eps_peak_true_um_m
    return c


def physical_diagnostics(df_te, yhat, coef):
    """Physical-admissibility diagnostics: peak-stress error, curve smoothness and the
    number of continuum-damage monotonicity violations (D = 1 - sigma/(E eps) must be
    non-decreasing with strain)."""
    d = df_te.copy(); d['pred'] = yhat
    pk, osc, dv = [], [], []
    for k, g in d.groupby(['nS', 'PVA']):
        g = g.sort_values('strain_pct')
        pk.append(abs(g.pred.max() - g.stress_MPa.max()) / g.stress_MPa.max() * 100.0)
        s = g.pred.values
        osc.append(np.mean(np.abs(np.diff(s, 2))) / (s.max() - s.min() + 1e-9))
        eps = g.eps_frac.values; m = eps >= 1.0e-3
        if m.sum() > 1:
            E = E_surface(coef, k[0], k[1])
            D = 1.0 - s[m] / (E * eps[m])
            dv.append(int(np.sum(np.diff(D) < -1e-6)))
        else:
            dv.append(0)
    return dict(peak_MAPE=float(np.mean(pk)), osc_index=float(np.mean(osc)),
                dmg_violations=float(np.mean(dv)))


def main():
    print('== data ==')
    df, pr, curves = build_dataset()
    df.to_csv(os.path.join(OUT, 'dataset_curves.csv'), index=False)
    pr.to_csv(os.path.join(OUT, 'peak_properties.csv'), index=False)
    print(df.shape, 'curves:', len(pr))

    # ---- Split A: random 80/20 over collocation points
    idx = RNG.permutation(len(df)); ntr = int(0.8 * len(df))
    A_tr, A_te = df.iloc[idx[:ntr]], df.iloc[idx[ntr:]]
    # ---- Split B: blind mixtures (5 of 24 mixtures fully held out)
    blind = df.apply(lambda r: (r.nS, r.PVA) in TEST_COMBOS, axis=1)
    B_te, B_tr = df[blind], df[~blind]
    print('split A', A_tr.shape, A_te.shape, '| split B', B_tr.shape, B_te.shape)

    coefA = fit_modulus_surface(pr, set(zip(A_tr.nS, A_tr.PVA)))
    coefB = fit_modulus_surface(pr, set(zip(B_tr.nS, B_tr.PVA)))

    results = []
    preds_store = {}
    calib_cache = {}
    Z = model_zoo(); print('models:', list(Z.keys()))

    for split, (tr, te), coef in [('A_random80-20', (A_tr, A_te), coefA),
                                  ('B_blind_mixtures', (B_tr, B_te), coefB)]:
        Xtr, ytr = tr[['nS', 'PVA', 'strain_pct']].values, tr.stress_MPa.values
        Xte, yte = te[['nS', 'PVA', 'strain_pct']].values, te.stress_MPa.values
        for name, mdl in Z.items():
            mdl.fit(Xtr, ytr)
            yh = mdl.predict(Xte)
            m = metrics(yte, yh)
            m.update(model=name, split=split, n_train=len(tr), n_test=len(te))
            results.append(m)
            preds_store[(split, name)] = yh
            print(f'  {split:18s} {name:26s} R2={m["R2"]:.4f} RMSE={m["RMSE"]:.3f}')
        # ---- physics-only Carreira-Chu
        yh_cc = np.full(len(te), np.nan)
        pos = {ix: i for i, ix in enumerate(te.index)}
        for k, g in te.groupby(['nS', 'PVA']):
            r0 = pr[(pr.nS == k[0]) & (pr.PVA == k[1])].iloc[0]
            e, s = carreira_chu(r0.sigma_peak, r0.strain_peak_pct / 100.0, r0.E0_GPa * 1000.0)
            yh_cc[[pos[i] for i in g.index]] = np.interp(g.strain_pct.values / 100.0, e, s)
        m = metrics(yte, yh_cc); m.update(model='Carreira-Chu (physics only)', split=split,
                                          n_train=0, n_test=len(te))
        results.append(m); preds_store[(split, 'Carreira-Chu (physics only)')] = yh_cc
        print(f'  {split:18s} {"Carreira-Chu (physics)":26s} R2={m["R2"]:.4f} RMSE={m["RMSE"]:.3f}')
        # ---- PINN family (physics weights tuned on an inner-validation split only)
        best = calibrate_pinn(tr, pr, coef); calib_cache[split] = best
        for tag, kw in [('PINN (physics-informed)', best),
                        ('NN data-only (no physics)', dict(lam_E=0.0, lam_dmg=0.0, lam_sm=0.0))]:
            yh = np.zeros(len(te))
            for seed in (0, 1, 2):                       # ensemble of 3 seeds
                p = train_pinn(tr, pr, coef, seed=seed, **kw)
                yh += p.predict(Xte, p.scX, p.scY)
            yh /= 3.0
            m = metrics(yte, yh); m.update(model=tag, split=split, n_train=len(tr), n_test=len(te))
            results.append(m); preds_store[(split, tag)] = yh
            print(f'  {split:18s} {tag:26s} R2={m["R2"]:.4f} RMSE={m["RMSE"]:.3f}')

    res = pd.DataFrame(results)[['split', 'model', 'n_train', 'n_test', 'R2', 'RMSE', 'MAE', 'MAPE']]
    res.to_csv(os.path.join(OUT, 'benchmark_metrics.csv'), index=False)

    # ---- physical-admissibility diagnostics for every model (blind split)
    diag = []
    for (sp, nm), yh in preds_store.items():
        if sp != 'B_blind_mixtures':
            continue
        dd = physical_diagnostics(B_te, yh, coefB); dd['model'] = nm; diag.append(dd)
    pd.DataFrame(diag).to_csv(os.path.join(OUT, 'physical_diagnostics.csv'), index=False)
    print(pd.DataFrame(diag).sort_values('R2', ascending=False).to_string() if False else
          pd.DataFrame(diag).to_string())

    # ---- per-mixture blind table
    rows = []
    for name in ['PINN (physics-informed)', 'Deep ANN (MLP 64-64-32)', 'Random Forest',
                 'Gaussian Process', 'SVR (RBF)', 'NN data-only (no physics)']:
        if ('B_blind_mixtures', name) in preds_store:
            c = curve_metrics(B_te, preds_store[('B_blind_mixtures', name)]); c['model'] = name
            rows.append(c)
    pd.concat(rows).to_csv(os.path.join(OUT, 'blind_per_mixture.csv'), index=False)

    # ---- ablation: physics weights
    abl = []
    XtrB, ytrB = B_tr[['nS', 'PVA', 'strain_pct']].values, B_tr.stress_MPa.values
    XteB, yteB = B_te[['nS', 'PVA', 'strain_pct']].values, B_te.stress_MPa.values
    bst = calib_cache.get('B_blind_mixtures') or calibrate_pinn(B_tr, pr, coefB)
    base = {k: bst[k] for k in ('lr', 'batch', 'epochs') if k in bst}
    for tag, kw in [('data only (no physics)', dict(base, lam_E=0, lam_dmg=0, lam_sm=0)),
                    ('+ elastic-modulus pin', dict(lam_E=bst['lam_E'], lam_dmg=0, lam_sm=0)),
                    ('+ damage consistency', dict(lam_E=bst['lam_E'], lam_dmg=bst['lam_dmg'], lam_sm=0)),
                    ('+ curvature smoothness (full PINN)', bst)]:
        yh = np.zeros(len(B_te))
        for seed in (0, 1, 2):
            m = train_pinn(B_tr, pr, coefB, seed=seed, **kw)
            yh += m.predict(XteB, m.scX, m.scY)
        yh /= 3.0
        mm = metrics(yteB, yh); mm['variant'] = tag
        mm.update(physical_diagnostics(B_te, yh, coefB))
        abl.append(mm)
    pd.DataFrame(abl).to_csv(os.path.join(OUT, 'ablation.csv'), index=False)
    print(pd.DataFrame(abl).to_string())

    # ---- feature importance (permutation) + peak-property LOOCV
    rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1).fit(XtrB, ytrB)
    pi = permutation_importance(rf, XteB, yteB, n_repeats=20, random_state=42)
    pd.DataFrame({'feature': ['nS', 'PVA', 'strain'], 'importance': pi.importances_mean,
                  'std': pi.importances_std}).to_csv(os.path.join(OUT, 'feature_importance.csv'), index=False)

    from sklearn.model_selection import LeaveOneOut
    loft = []
    for target in ['sigma_peak', 'strain_peak_um_m']:
        X = pr[['nS', 'PVA']].values; y = pr[target].values
        for nm, mdl in [('GPR', make_pipeline(StandardScaler(), GaussianProcessRegressor(
                            kernel=C(1.0) * RBF([1.0, 1.0]) + WhiteKernel(1e-3), normalize_y=True, random_state=42))),
                        ('Random Forest', RandomForestRegressor(n_estimators=400, random_state=42)),
                        ('Quadratic Reg.', make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1e-3)))]:
            yh = np.zeros_like(y)
            for tr_i, te_i in LeaveOneOut().split(X):
                mdl.fit(X[tr_i], y[tr_i]); yh[te_i] = mdl.predict(X[te_i])
            mm = metrics(y, yh); mm.update(target=target, model=nm)
            loft.append(mm)
    pd.DataFrame(loft).to_csv(os.path.join(OUT, 'peak_loocv.csv'), index=False)

    json.dump({'test_combos': TEST_COMBOS,
               'n_points': int(len(df)), 'n_mixtures': int(len(pr)),
               'splitA': [int(len(A_tr)), int(len(A_te))], 'splitB': [int(len(B_tr)), int(len(B_te))],
               'coefA': list(map(float, coefA)), 'coefB': list(map(float, coefB))},
              open(os.path.join(OUT, 'summary.json'), 'w'), indent=2)
    np.savez(os.path.join(OUT, 'predictions.npz'),
             **{f'{k[0]}||{k[1]}'.replace(' ', '_'): v for k, v in preds_store.items()})
    print('done')


if __name__ == '__main__':
    main()
