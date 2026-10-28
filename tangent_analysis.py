#!/usr/bin/env python3
"""Reproduce Table 17 of the manuscript: tangent-modulus diagnostics.

The manuscript argues that a constitutive surrogate must supply a usable tangent
d sigma / d epsilon for an implicit (Newton-Raphson) solver, and that $R^2$ does
not reveal whether it does.  This script quantifies three quantities that the
100 microstrain grid can actually support:

  E0        through-origin regression of stress on strain below 0.6 sigma_p
            (the initial tangent modulus), reported in GPa
  E_sec     secant modulus at peak, sigma_p / epsilon_p, in GPa
  roughness total variation of the discrete derivative normalised by
            (|mean derivative| * number of intervals).  A smooth constitutive
            tangent gives a small value; an oscillating derivative gives a
            large one.

The instantaneous softening tangent is deliberately NOT reported: at ~31
collocation points per curve the grid cannot resolve it (see Table 3 and its
caption in the manuscript).

Input   results/preds2.npz  (blind-mixture predictions, all backbones)
Output  results/table17_tangent.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, 'results', 'preds2.npz')
OUT = os.path.join(ROOT, 'results', 'table17_tangent.csv')

# backbone key in the archive -> label used in the manuscript
MODELS = {
    'MLP data-only ($\\lambda=0$)': 'MLP, data only (lambda=0)',
    'MLP-PINN ($\\lambda=1,1,1$)': 'MLP, full physics (lambda=1,1,1)',
    'PIKAN (data only)': 'Spline network, data only',
    'PIKAN (elastic pin on)': 'Spline network, elastic pin',
    'DeepONet': 'DeepONet',
    'PINO (elastic pin on)': 'DeepONet, elastic pin',
    'XGBoost': 'Gradient-boosted tree',
}


def metrics(sigma, eps):
    """E0 [GPa], secant at peak [GPa], derivative roughness [-]."""
    order = np.argsort(eps)
    sigma, eps = sigma[order], eps[order]
    ip = int(np.argmax(sigma))
    sp, ep = sigma[ip], eps[ip]

    m = (eps <= ep) & (sigma <= 0.6 * sp) & (sigma > 0)
    e0 = (sigma[m] @ eps[m]) / (eps[m] @ eps[m]) * 1e3 if m.sum() >= 3 else np.nan
    esec = sp / ep * 1e3 if ep > 0 else np.nan

    d = np.diff(sigma) / np.diff(eps)
    d = d[np.isfinite(d)]
    if len(d) > 2 and np.mean(d) != 0:
        rough = float(np.sum(np.abs(np.diff(d))) / (abs(np.mean(d)) * len(d)))
    else:
        rough = np.nan
    return e0, esec, rough


def collect(sigma, ns, pva, eps):
    combos = sorted(set(zip(ns, pva)))
    rows = []
    for c in combos:
        k = (ns == c[0]) & (pva == c[1])
        rows.append(metrics(sigma[k], eps[k]))
    return np.array(rows, float), combos


def main():
    z = np.load(NPZ, allow_pickle=True)
    Xq, yq = z['Xq'], z['yq']
    ns, pva, eps = Xq[:, 0], Xq[:, 1], Xq[:, 2] * 1e4      # strain_pct -> microstrain

    truth, combos = collect(yq, ns, pva, eps)
    print('blind mixtures:', [f'{a:g}%nS/{b:g}%PVA' for a, b in combos])
    print('measured roughness: %.2f\n' % np.nanmean(truth[:, 2]))

    out = [dict(model='Measured curve', E0_error_pct=np.nan,
                secant_error_pct=np.nan, roughness=np.nanmean(truth[:, 2]))]
    for key, label in MODELS.items():
        if key not in z.files:
            print('  (missing in archive: %s)' % key)
            continue
        pred, _ = collect(z[key], ns, pva, eps)
        rel = np.abs(pred - truth) / np.abs(truth) * 100
        out.append(dict(model=label,
                        E0_error_pct=round(float(np.nanmean(rel[:, 0])), 1),
                        secant_error_pct=round(float(np.nanmean(rel[:, 1])), 1),
                        roughness=round(float(np.nanmean(pred[:, 2])), 2)))

    df = pd.DataFrame(out)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print('\nwrote', os.path.relpath(OUT, ROOT))


if __name__ == '__main__':
    main()
