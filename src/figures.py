#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the ECC ML/DL/PINN surrogate benchmark.

File names follow the FINAL reading order of the manuscript:
  fig01 experimental curves          fig07 accuracy vs physical admissibility
  fig02 PINN architecture            fig08 ablation of the physics terms
  fig03 predicted vs experimental    fig09 digital-twin output
  fig04 benchmark bars (blind)       fig10 GP surrogate with uncertainty
  fig05 blind-mixture curves         fig11 response surface
  fig06 parity plots (blind)         fig12 workflow (Appendix A)
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (build_dataset, fit_modulus_surface, train_pinn,
                      TEST_COMBOS, PVA_LEVELS)

ROOT = os.path.expanduser('~/workspace/paper')
OUT = os.path.join(ROOT, 'out'); FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'figure.dpi': 110, 'savefig.bbox': 'tight',
    # embed TrueType (never Type-3 bitmaps) so every PDF viewer renders the figures
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    # --- TikZ/PGF backend: text is typeset by LaTeX, inheriting Palatino
    'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
    'pgf.preamble': '\n'.join([r'\usepackage[T1]{fontenc}',
                               r'\usepackage{mathpazo}',
                               r'\usepackage{amsmath}',
                               r'\usepackage{amssymb}'])})

CPVA = {0.5: '#0B3C5D', 1.0: '#2E86AB', 1.5: '#F4A259', 2.0: '#C1272D'}
ABL_COLORS = ['#8C8C8C', '#2E86AB', '#1B998B', '#C1272D']     # shared with fig07/fig08
FAM_COLOR = {'Constitutive / hybrid': ABL_COLORS[0], 'Machine learning': ABL_COLORS[1],
             'Deep learning': ABL_COLORS[2], 'PINN (proposed)': ABL_COLORS[3]}


def fam(m):
    if m.startswith('PINN'):
        return 'PINN (proposed)'
    if 'no physics' in m or m.startswith('Deep ANN') or m.startswith('ANN ('):
        return 'Deep learning'
    if 'physics only' in m or 'Hybrid' in m:
        return 'Constitutive / hybrid'
    return 'Machine learning'


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=300)
    # native TikZ/PGF export: every drawing is embedded as vector LaTeX code
    try:
        from matplotlib.backends.backend_pgf import FigureCanvasPgf
        fig.set_canvas(FigureCanvasPgf(fig))
        fig.savefig(os.path.join(FIG, f'{name}.pgf'), format='pgf')
    except Exception as exc:
        print('  PGF FAILED', name, type(exc).__name__, exc)
    plt.close(fig); print('  saved', name, '(png+pdf+pgf)')


# ------------------------------------------------------------------ Figure 1
def fig01_curves(curves):
    # PVA levels reuse the four-colour palette of Figure 9(a)
    _pal = dict(zip(sorted({p for _, p in curves}), ABL_COLORS))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.0), sharex=True, sharey=True)
    for ax, ns in zip(axes.ravel(), sorted({n for n, _ in curves})):
        for (n2, pva), pts in sorted(curves.items()):
            if n2 != ns:
                continue
            ax.plot(pts[:, 0], pts[:, 1], '-', color=_pal[pva], lw=1.2, label=f'{pva:.1f}%')
        ax.set_title(f'$n_S$ = {ns:.0f}%', fontsize=9)
    for ax in axes.ravel():
        ax.set_xlim(0, 3.4); ax.set_ylim(0, 85)
        ax.set_xlabel('Strain ($\\times10^{3}\\,\\mu\\varepsilon$)')
        ax.set_ylabel('Stress (MPa)')
    axes[0, 0].legend(fontsize=7, loc='lower right', title='PVA', title_fontsize=7)
    save(fig, 'fig01_experimental_curves')


# ------------------------------------------------------------------ Figure 2
def fig02_architecture():
    """Compact architecture diagram: network, the four loss terms (the three physics
    residuals sit on a tinted band with no bounding boxes), the composite objective
    and the training loop."""
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off'); ax.grid(False)

    def bx(x0, y0, x1, y1, fc, ec='#54606b', lw=0.9):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                     boxstyle='round,pad=0.3,rounding_size=0.9', fc=fc, ec=ec, lw=lw))

    def arrow(x0, y0, x1, y1, ls='-', lw=0.9, color='#38434d'):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8,
                                     lw=lw, ls=ls, color=color, shrinkA=0, shrinkB=0))

    # ---- network (no separator rule line) --------------------------------
    cin, ch1, ch2, ch3, cout = 11.0, 30.0, 48.0, 66.0, 83.0
    in_y = [70.0, 76.0, 82.0]
    h_y = list(np.linspace(66.0, 88.0, 7))
    col = {cin: ABL_COLORS[0], ch1: ABL_COLORS[1], ch2: ABL_COLORS[1],
       ch3: ABL_COLORS[2], cout: ABL_COLORS[3]}
    nodes = {cin: in_y, ch1: h_y, ch2: h_y, ch3: h_y, cout: [77.0]}
    for a, b in ((cin, ch1), (ch1, ch2), (ch2, ch3), (ch3, cout)):
        for ya in nodes[a]:
            for yb in nodes[b]:
                ax.plot([a, b], [ya, yb], color='#8496a8', lw=0.28, alpha=0.13, zorder=1)
    for x in (cin, ch1, ch2, ch3, cout):
        r = 1.6 if x == cin else (1.7 if x == cout else 1.05)
        for yy in nodes[x]:
            ax.add_patch(plt.Circle((x, yy), r, fc=col[x], ec='none', zorder=3, alpha=0.95))
    for yy, lab in zip(in_y, ['$n_S$ (%)', '$PVA$ (%)', '$\\varepsilon$ (%)']):
        ax.text(cin - 2.6, yy, lab, ha='right', va='center', fontsize=7.2, zorder=4)
    ax.text(cout + 2.8, 77.0, '$\\hat{\\sigma}$ (MPa)', ha='left', va='center', fontsize=7.2, zorder=4)
    for x, t in [(cin, 'Input\n3 features'), (ch1, 'FC 1\n64 units'), (ch2, 'FC 2\n64 units'),
                 (ch3, 'FC 3\n32 units'), (cout, 'Output\nlinear')]:
        ax.text(x, 94.5, t, ha='center', va='center', fontsize=6.6, linespacing=1.3, zorder=4)
    ax.text(50, 61.5, 'hidden units: hyper-tangent activation', ha='center',
            fontsize=6.4, style='italic', color='#4a5560')

    # ---- distribution bus ------------------------------------------------
    ax.plot([48, 48], [60.0, 58.0], color='#38434d', lw=0.9)
    ax.plot([8, 92], [58.0, 58.0], color='#38434d', lw=0.9)
    for xb in (19, 43, 64, 86):
        arrow(xb, 58.0, xb, 55.8)

    # ---- tinted band grouping the three PHYSICS residuals (no boxes) -----
    ax.add_patch(FancyBboxPatch((32.0, 41.5), 66.0, 15.5, boxstyle='round,pad=0.3,rounding_size=0.9',
                                fc='#F5F8FA', ec='none', zorder=0))
    ax.text(67.0, 59.6, 'physics residuals  (added to the data misfit)', ha='center',
            fontsize=6.6, style='italic', color='#4a5560')

    # ---- data-misfit term (keeps its box: it is the data term) -----------
    bx(6.0, 41.5, 29.0, 57.0, '#E8F1F8')
    ax.text(17.5, 54.4, 'Data misfit  $\\mathcal{L}_{data}$', ha='center', fontsize=7.0,
            fontweight='bold')
    ax.text(17.5, 47.8, '$(1/N)\\sum_i(\\hat\\sigma_i-\\sigma_i)^2$\nover the $N$ training\n'
                        'collocation points', ha='center', va='center', fontsize=6.1, linespacing=1.5)

    # ---- three physics terms: plain text, no bounding box ----------------
    terms = [
        (43.0, 'Elastic-modulus pin  $\\mathcal{L}_{E}$',
         '$\\partial\\hat\\sigma/\\partial\\varepsilon=\\hat{E}(n_S,PVA)$\n'
         'as $\\varepsilon\\to0$  (central difference)\nbranch $\\varepsilon\\leq600$ $\\mu\\varepsilon$'),
        (64.0, 'Damage consistency  $\\mathcal{L}_{D}$',
         '$D=1-\\hat\\sigma/(\\hat{E}\\varepsilon)$\nbounded in $[0,1]$; monotonicity\nnot enforced\n'
         'branch $\\varepsilon\\geq1000$ $\\mu\\varepsilon$'),
        (86.0, 'Curvature smoothness  $\\mathcal{L}_{S}$',
         '$\\sum_i[\\hat\\sigma_{i+1}-2\\hat\\sigma_i+\\hat\\sigma_{i-1}]^2$\n'
         'single peak, no oscillation\nfull collocation grid'),
    ]
    for xc, title, body in terms:
        ax.text(xc, 54.4, title, ha='center', fontsize=7.0, fontweight='bold')
        ax.text(xc, 47.6, body, ha='center', va='center', fontsize=6.1, linespacing=1.55)

    for xb in (19, 43, 64, 86):
        arrow(xb, 41.5, 50, 37.6)

    bx(22.0, 27.0, 78.0, 37.6, '#FFFFFF', lw=1.2)
    ax.text(50, 32.3, r'$\mathcal{L}(\theta)=\mathcal{L}_{data}+\lambda_E\mathcal{L}_{E}'
                      r'+\lambda_D\mathcal{L}_{D}+\lambda_S\mathcal{L}_{S}$',
            ha='center', va='center', fontsize=9.0)
    arrow(50, 27.0, 50, 24.6)
    bx(12.0, 13.0, 88.0, 24.6, '#EDEDED')
    ax.text(50, 21.3, 'Adam:  lr $10^{-2}$, step decay $\\times0.6$ / 3000 iters, '
                      '12 000 iterations, mini-batch 64', ha='center', va='center', fontsize=6.5)
    ax.text(50, 17.3, 'three-seed ensemble  $\\to$  mean prediction is the surrogate output',
            ha='center', va='center', fontsize=6.5)
    ax.text(50, 7.4, '$\\lambda_E,\\ \\lambda_D,\\ \\lambda_S$ selected on an inner validation split '
                     'carved from the training mixtures; the blind test mixtures are never used',
            ha='center', va='center', fontsize=6.3, color='#8a2b23')

    # ---- training feedback loop, routed clear of every box ---------------
    ax.add_patch(FancyArrowPatch((12.0, 15.4), (1.5, 15.4), arrowstyle='-', lw=0.8,
                                 color='#8a97a3', ls=':'))
    ax.add_patch(FancyArrowPatch((1.5, 15.4), (1.5, 91.5), arrowstyle='-', lw=0.8,
                                 color='#8a97a3', ls=':'))
    ax.add_patch(FancyArrowPatch((1.5, 91.5), (9.5, 91.5), arrowstyle='-|>',
                                 mutation_scale=8, lw=0.8, color='#8a97a3', ls=':'))
    ax.text(3.3, 42.0, 'gradient updates', rotation=90, fontsize=5.9, color='#8a97a3',
            va='center', ha='center')
    save(fig, 'fig03_pinn_architecture')


def fig03_predicted_ns():
    """Predicted versus experimental curves for the six nanosilica contents, one row per
    surrogate family (ML, DL, physics-informed).  Models were trained on the blind-split
    training set, so the five blind compositions are genuine extrapolations."""
    z = np.load(os.path.join(OUT, 'pred_all.npz'), allow_pickle=True)
    nS = z['dfall_ns']; PVA = z['dfall_pva']; STR = z['dfall_strain_pct']; SIG = z['dfall_stress']
    models = ['XGBoost', 'Deep ANN', 'PINN']
    pretty = {'XGBoost': 'XGBoost  (ML)', 'Deep ANN': 'Deep ANN  (DL)', 'PINN': 'PINN  (proposed)'}

    fig, axes = plt.subplots(3, 6, figsize=(16.0, 8.2), sharex=True)
    for r, mname in enumerate(models):
        pred = z[mname]
        for c, ns in enumerate(range(6)):
            ax = axes[r, c]
            for pva in PVA_LEVELS:
                g = np.where((nS == ns) & (PVA == pva))[0]
                if g.size == 0:
                    continue
                o = g[np.argsort(STR[g])]
                ax.plot(STR[o] * 10, SIG[o], '-', color=CPVA[pva], lw=1.15, alpha=0.95)
                ax.plot(STR[o] * 10, pred[o], '--', color=CPVA[pva], lw=1.0, alpha=0.95)
                if (ns, pva) in TEST_COMBOS:
                    ax.plot(STR[o][np.argmax(pred[o])] * 10, pred[o].max(), '*',
                            color=CPVA[pva], ms=9, mec='k', mew=0.4, zorder=5)
            yt = SIG[np.where(nS == ns)[0]]; yp = pred[np.where(nS == ns)[0]]
            r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
            ax.text(0.045, 0.94, f'$R^2$ = {r2:.3f}', transform=ax.transAxes,
                    fontsize=6.4, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#c8d2dc', lw=0.5, alpha=0.9))
            if r == 0:
                ax.set_title(f'$n_S$ = {ns}%', fontsize=9)
            if c == 0:
                ax.set_ylabel(f'{pretty[mname]}\nStress (MPa)', fontsize=7.4)
            if r == 2:
                ax.set_xlabel('Strain ($\\times10^{3}\\,\\mu\\varepsilon$)', fontsize=7.4)
            ax.set_xlim(0, 3.45); ax.set_ylim(0, 88)
            ax.tick_params(labelsize=6.5)

    h_exp = [plt.Line2D([], [], color='#444', lw=1.3) for _ in range(1)]
    h_pred = [plt.Line2D([], [], color='#444', lw=1.3, ls='--')]
    h_star = [plt.Line2D([], [], color='#444', marker='*', ls='', ms=9, mec='k', mew=0.4)]
    hs = [plt.Line2D([], [], color=CPVA[p], lw=2) for p in PVA_LEVELS] + h_exp + h_pred + h_star
    lb = [f'PVA {p:.1f}%' for p in PVA_LEVELS] + \
         ['experiment', 'surrogate prediction', 'blind composition']
    fig.legend(hs, lb, ncol=7, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])
    save(fig, 'fig04_predicted_ns_curves')


# ------------------------------------------------------------------ Figure 4
def fig04_bars(res, diag):
    d = res[res.split == 'B_blind_mixtures'].merge(diag[['model', 'peak_MAPE']],
                                                   on='model', how='left').sort_values('R2')
    pinn = float(d.loc[d.model.str.startswith('PINN'), 'R2'].iloc[0])
    xref, xlab = 1.0, 1.035
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.barh(d.model, d.R2, color=[FAM_COLOR[fam(m)] for m in d.model], height=0.70, zorder=3)
    ax.axvline(xref, color='#5b6770', ls='--', lw=0.9, zorder=4)
    ax.axvline(pinn, color='#C1272D', ls=':', lw=1.1, zorder=4)
    for m, v, r in zip(d.model, d.R2, d.RMSE):
        ax.text(xlab, m, f'{v:.4f}   {r:5.2f} MPa', va='center', ha='left', fontsize=6.8, zorder=5)
    ax.text(xlab, len(d) - 0.25, '$R^2$        RMSE', va='center', ha='left',
            fontsize=6.6, style='italic', color='#444', zorder=5)
    ax.set_xlim(0, 1.34); ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel('$R^2$ on the five blind mixtures')
    ax.tick_params(axis='y', labelsize=7.4)
    ax.axvspan(xref, 1.34, color='#000000', alpha=0.03, zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in FAM_COLOR.values()]
    ax.legend(handles, list(FAM_COLOR.keys()), fontsize=6.8, ncol=4, loc='upper center',
              bbox_to_anchor=(0.5, -0.085), frameon=False, title='model family',
              title_fontsize=6.8, columnspacing=1.5, handlelength=1.4)
    fig.tight_layout()
    save(fig, 'fig05_benchmark_bars')


# ------------------------------------------------------------------ Figure 5
def fig05_blind(df, preds, test_combos):
    names = [(FAM_COLOR['PINN (proposed)'], '-', 'PINN (proposed)'),
             ('#B0392E', '-.', 'Hybrid PIML'),
             (FAM_COLOR['Machine learning'], '--', 'XGBoost'),
             (FAM_COLOR['Deep learning'], ':', 'NN (data only)')]
    fig, axes = plt.subplots(1, len(test_combos), figsize=(11.6, 3.1), sharey=False)
    idx = preds[('B_blind_mixtures', '__idx__')]
    for ax, key in zip(axes, test_combos):
        g = df[(df.nS == key[0]) & (df.PVA == key[1])].sort_values('strain_pct')
        ax.plot(g.strain_pct * 10, g.stress_MPa, 'o', ms=2.6, color='#333', label='Experiment')
        for col, ls, nm in names:
            k = ('B_blind_mixtures', nm)
            if k not in preds:
                continue
            m = np.isin(idx, g.index.values)
            ax.plot(g.strain_pct * 10, preds[k][m], ls, color=col, lw=1.3, label=nm)
        ax.set_title(f'$n_S$ {key[0]:.0f}% / PVA {key[1]:.1f}%', fontsize=8.5)
        ax.set_xlabel('Strain ($\\times10^{3}\\,\\mu\\varepsilon$)')
    axes[0].set_ylabel('Stress (MPa)')
    axes[1].legend(fontsize=6.4, loc='lower right')
    save(fig, 'fig06_blind_curves')


# ------------------------------------------------------------------ Figure 6
def fig06_parity(df_te, preds, models):
    n = len(models); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.0, 12.4))
    idx = preds[('B_blind_mixtures', '__idx__')]
    y = df_te.loc[idx, 'stress_MPa'].values
    for ax, nm in zip(axes.ravel(), models):
        yh = preds[('B_blind_mixtures', nm)]
        ax.scatter(y, yh, s=4, alpha=1.0, color=FAM_COLOR[fam(nm)], edgecolors='none')
        lim = [0, max(y.max(), yh.max()) * 1.05]
        ax.plot(lim, lim, 'k--', lw=.7)
        r2 = 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
        ax.set_title(f'{nm}\n$R^2$ = {r2:.4f}', fontsize=7)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.tick_params(labelsize=6.5)
        ax.set_xlabel('Measured (MPa)', fontsize=7); ax.set_ylabel('Predicted (MPa)', fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.tight_layout()
    save(fig, 'fig07_parity_blind')


# ------------------------------------------------------------------ Figure 7
def fig08_accuracy_vs_physics(res, diag):
    d = res[res.split == 'B_blind_mixtures'].merge(diag, on='model', how='inner')
    d = d[d.model.str.contains('Hybrid') == False].sort_values('R2', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    # each model name sits immediately beside its own marker: no leader lines.
    # Labels whose horizontal spans overlap are separated vertically by MINSEP
    # so the tight R^2 cluster stays legible.
    MINSEP, MW, XOFF, LABW = 0.0036, 0.0024, 0.40, 2.85
    mk = [(float(r.dmg_violations), float(r.R2)) for _, r in d.iterrows()]
    placed = []

    def clear(x0, x1, y):
        for mx, my in mk:
            if x0 - 0.06 <= mx <= x1 and abs(y - my) < MW:
                return False
        for a, b, yp in placed:
            if not (x1 < a or x0 > b) and abs(y - yp) < MINSEP:
                return False
        return True

    for _, r in d.sort_values('R2', ascending=False).iterrows():
        c = FAM_COLOR[fam(r.model)]
        isp = r.model.startswith('PINN')
        ax.scatter([r.dmg_violations], [r.R2], s=140 if isp else 34, marker='*' if isp else 'o',
                   color=c, zorder=3, edgecolors='none')
        choice = None
        for side in ('r', 'l'):
            if side == 'r':
                x0 = r.dmg_violations + XOFF; x1 = x0 + LABW; ha = 'left'
            else:
                x1 = r.dmg_violations - XOFF; x0 = x1 - LABW; ha = 'right'
            for k in range(46):
                for sgn in ((0,) if k == 0 else (1, -1)):
                    if clear(x0, x1, r.R2 + sgn * k * 0.0007):
                        choice = (x0, x1, r.R2 + sgn * k * 0.0007, ha); break
                if choice:
                    break
            if choice:
                break
        if choice is None:
            x0 = r.dmg_violations + XOFF; choice = (x0, x0 + LABW, r.R2, 'left')
        x0, x1, y, ha = choice
        ax.text(x0, y, r.model, fontsize=6.3, va='center', ha=ha, color='#333', zorder=4)
        placed.append((x0, x1, y))
    ax.set_xlim(-0.4, 13.6); ax.set_ylim(0.886, 1.004)
    ax.set_yticks(np.arange(0.90, 1.001, 0.02))
    ax.set_xlabel('damage-monotonicity violations per curve')
    ax.set_ylabel('$R^2$ (blind mixtures)')
    ax.axvspan(6.8, 13.6, color='#000000', alpha=0.025, zorder=0)
    handles = [plt.Line2D([], [], marker='o', ls='', ms=5, color=c) for c in FAM_COLOR.values()]
    # legend placed BELOW the axes: every in-axes corner holds data points
    ax.legend(handles, list(FAM_COLOR.keys()), fontsize=7, ncol=4, loc='upper center',
              bbox_to_anchor=(0.5, -0.105), frameon=False, title='model family',
              title_fontsize=7, columnspacing=1.5, handlelength=1.3)
    fig.tight_layout()
    save(fig, 'fig08_accuracy_vs_physics')


# ------------------------------------------------------------------ Figure 8
def fig09_ablation(abl):
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.1))
    x = np.arange(len(abl))
    labels = ['none', '$+\\mathcal{L}_E$', '$+\\mathcal{L}_E\\mathcal{L}_D$',
              '$+\\mathcal{L}_E\\mathcal{L}_D\\mathcal{L}_S$']
    for ax, (col, lab) in zip(axes, [('R2', '$R^2$'), ('RMSE', 'RMSE (MPa)'),
                                     ('osc_index', 'oscillation index (-)'),
                                     ('dmg_violations', 'damage violations / curve')]):
        ax.bar(x, abl[col].values, color=ABL_COLORS)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(lab); ax.set_title(lab, fontsize=9)
    fig.tight_layout()
    save(fig, 'fig09_ablation')


# ------------------------------------------------------------------ Figure 9
def fig09_digitaltwin(dfall, pr):
    coef = fit_modulus_surface(pr)
    m = train_pinn(dfall, pr, coef, seed=0, lam_E=0.0, lam_dmg=0.0, lam_sm=0.0)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    for i, ns in enumerate(np.arange(0.5, 5.1, 0.5)):
        eps = np.linspace(0, 0.30, 60)
        X = np.column_stack([np.full(60, ns), np.full(60, 1.0), eps])
        ax.plot(eps * 10, m.predict(X, m.scX, m.scY), color=plt.cm.viridis(i / 9), lw=1.5,
                label=f'{ns:.1f}')
    ax.set_xlabel('Strain ($\\times10^{3}\\,\\mu\\varepsilon$)'); ax.set_ylabel('Stress (MPa)')
    ax.set_title('(a) instant curves for unseen $n_S$ (PVA = 1.0%)', fontsize=9)
    ax.legend(fontsize=6, title='$n_S$ (%)', title_fontsize=6, ncol=2)
    ax2 = axes[1]
    nsg = np.linspace(0, 5, 70); pvg = np.linspace(0.5, 2.0, 60)
    NSg, PVg = np.meshgrid(nsg, pvg); Zp = np.zeros_like(NSg)
    eps = np.linspace(0, 0.30, 60)
    for i in range(NSg.shape[0]):
        for j in range(NSg.shape[1]):
            X = np.column_stack([np.full(60, NSg[i, j]), np.full(60, PVg[i, j]), eps])
            Zp[i, j] = m.predict(X, m.scX, m.scY).max()
    cs = ax2.contourf(NSg, PVg, Zp, levels=18, cmap='magma')
    ax2.plot(*zip(*TEST_COMBOS), 'w*', ms=12, mec='k', label='blind mixtures')
    ax2.set_xlabel('Nanosilica (%)'); ax2.set_ylabel('PVA (%)')
    ax2.set_title('(b) predicted peak strength map', fontsize=9)
    ax2.legend(fontsize=6.5, loc='lower right')
    plt.colorbar(cs, ax=ax2, label='$\\sigma_{peak}$ (MPa)')
    fig.tight_layout()
    save(fig, 'fig10_digital_twin')


# ------------------------------------------------------------------ Figure 10
def fig11_uncertainty(df, pr, B_tr, B_te):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = sc.fit_transform(B_tr[['nS', 'PVA', 'strain_pct']].values)
    gp = GaussianProcessRegressor(kernel=Ck(1.0) * RBF([1., 1., 1.]) + WhiteKernel(1e-3),
                                  normalize_y=True, random_state=42).fit(Xtr, B_tr.stress_MPa.values)
    fig, axes = plt.subplots(1, len(TEST_COMBOS), figsize=(11.6, 2.8), sharey=False)
    for ax, key in zip(axes, TEST_COMBOS):
        g = df[(df.nS == key[0]) & (df.PVA == key[1])].sort_values('strain_pct')
        Xte = sc.transform(g[['nS', 'PVA', 'strain_pct']].values)
        mu, sd = gp.predict(Xte, return_std=True)
        ax.plot(g.strain_pct * 10, g.stress_MPa, 'o', ms=2.6, color='#333', label='Experiment')
        ax.plot(g.strain_pct * 10, mu, '-', color=FAM_COLOR['PINN (proposed)'], lw=1.3, label='GP mean')
        ax.fill_between(g.strain_pct * 10, mu - 2 * sd, mu + 2 * sd,
                        color=FAM_COLOR['PINN (proposed)'], alpha=.18, label='$\\pm2\\sigma$')
        ax.set_title(f'$n_S$ {key[0]:.0f}% / PVA {key[1]:.1f}%', fontsize=8.5)
        ax.set_xlabel('Strain ($\\times10^{3}\\,\\mu\\varepsilon$)')
    axes[0].set_ylabel('Stress (MPa)')
    axes[2].legend(fontsize=6.4, loc='lower right')
    save(fig, 'fig11_uncertainty')


# ------------------------------------------------------------------ Figure 11
def fig11_surface(dfall):
    from sklearn.ensemble import RandomForestRegressor
    X = dfall[['nS', 'PVA', 'strain_pct']].values; y = dfall.stress_MPa.values
    rf = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1).fit(X, y)
    ns = np.linspace(0, 5, 60); eps = np.linspace(0, 0.32, 70)
    NS, EP = np.meshgrid(ns, eps)
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
    for ax, pva in zip(axes, PVA_LEVELS):
        Z = rf.predict(np.column_stack([NS.ravel(), np.full(NS.size, pva), EP.ravel()])).reshape(NS.shape)
        cs = ax.contourf(NS, EP * 10, Z, levels=18, cmap='viridis')
        ax.set_title(f'PVA = {pva:.1f}%', fontsize=8.5)
        ax.set_xlabel('$n_S$ (%)')
        plt.colorbar(cs, ax=ax, label='$\\sigma$ (MPa)' if pva == 0.5 else None, shrink=.9)
    axes[0].set_ylabel('Strain ($\\times10^{3}\\,\\mu\\varepsilon$)')
    fig.tight_layout()
    save(fig, 'fig12_response_surface')


# ------------------------------------------------------------------ Figure 12
def fig02_workflow():
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 49.2); ax.axis('off'); ax.grid(False)
    W, H, GAP = 15.0, 12.0, 1.6
    x0 = (100 - (6 * W + 5 * GAP)) / 2.0
    top = 36.0
    boxes = [
        ('Experimental\nprogramme', '24 nS/PVA ECC\nstress–strain curves', '#E8F1F8'),
        ('Collocation\ndataset', '$[n_S,\\,PVA,\\,\\varepsilon]\\to\\sigma$\n737 points', '#E8F1F8'),
        ('Splitting', 'random 80/20\n$+$ blind mixture\nhold-out', '#FDECEA'),
        ('Surrogate\nmodels', 'Data-driven and\nphysics-informed model\nfamilies (see Table 4)', '#EAF6F1'),
        ('Blind\nassessment', '$R^2$, RMSE, peak\nerror, oscillation', '#FDECEA'),
        ('Digital twin', 'instant curves\nfor new mixes', '#FFF6E5'),
    ]
    xs = []
    for i, (title, body, fc) in enumerate(boxes):
        x = x0 + i * (W + GAP); xs.append(x)
        ax.add_patch(FancyBboxPatch((x, top), W, H, boxstyle='round,pad=0.35,rounding_size=1.2',
                                    fc=fc, ec='#54606b', lw=0.9))
        ax.text(x + W / 2, top + H - 3.1, title, ha='center', va='center',
                fontsize=8.2, fontweight='bold', linespacing=1.25)
        ax.text(x + W / 2, top + 4.0, body, ha='center', va='center', fontsize=7.0, linespacing=1.45)
    for i in range(len(xs) - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + W + 0.15, top + H / 2), (xs[i + 1] - 0.15, top + H / 2),
                                     arrowstyle='-|>', mutation_scale=10, lw=1.0, color='#38434d'))
    # keep the objective box inside the axes (previously 48.8 -> 101.0,
    # so its right edge fell outside x=100 and was clipped)
    dx0, dx1 = 26.0, 92.0
    ax.add_patch(FancyBboxPatch((dx0, 12.5), dx1 - dx0, 11.5, boxstyle='round,pad=0.35,rounding_size=1.2',
                                fc='#F4F7FA', ec='#54606b', lw=0.9))
    ax.text((dx0 + dx1) / 2, 21.2, 'Physics-informed objective', ha='center',
            fontsize=8.0, fontweight='bold')
    ax.text((dx0 + dx1) / 2, 15.6,
            r'$\mathcal{L}=\mathcal{L}_{data}+\lambda_E\mathcal{L}_{E}+\lambda_D\mathcal{L}_{D}+\lambda_S\mathcal{L}_{S}$'
            r'     ($\mathcal{L}_E$ elastic pin, $\mathcal{L}_D$ damage consistency, $\mathcal{L}_S$ smoothness)',
            ha='center', va='center', fontsize=7.0)
    ax.add_patch(FancyArrowPatch((xs[3] + W / 2, top - 0.3), (xs[3] + W / 2, 24.0),
                                 arrowstyle='-|>', mutation_scale=10, lw=1.0, color='#38434d'))
    save(fig, 'fig02_workflow')


# ------------------------------------------------------------------------- main
def main():
    dfall, pr, curves = build_dataset()
    res = pd.read_csv(os.path.join(OUT, 'benchmark_metrics.csv'))
    abl = pd.read_csv(os.path.join(OUT, 'ablation.csv'))
    diag = pd.read_csv(os.path.join(OUT, 'physical_diagnostics.csv'))
    z = np.load(os.path.join(OUT, 'predictions.npz'), allow_pickle=True)
    # the archive stores keys as 'split||model_name' with spaces turned into
    # underscores; map them back so display names match the result tables
    preds = {}
    for _k in z.files:
        _sp, _nm = _k.split('||')
        preds[(_sp, _nm.replace('_', ' '))] = z[_k]
    blind = dfall.apply(lambda r: (r.nS, r.PVA) in TEST_COMBOS, axis=1)
    B_te, B_tr = dfall[blind], dfall[~blind]
    preds[('B_blind_mixtures', '__idx__')] = B_te.index.values

    print('figures...')
    fig01_curves(curves)
    fig02_architecture()
    fig03_predicted_ns()
    fig04_bars(res, diag)
    fig05_blind(dfall, preds, TEST_COMBOS)
    models = [m for m in res[res.split == 'B_blind_mixtures'].sort_values('R2', ascending=False).model
              if ('B_blind_mixtures', m) in preds]
    fig06_parity(B_te, preds, models[:16])
    fig08_accuracy_vs_physics(res, diag)
    fig09_ablation(abl)
    fig09_digitaltwin(dfall, pr)
    fig11_uncertainty(dfall, pr, B_tr, B_te)
    fig11_surface(dfall)
    fig02_workflow()
    print('all figures done')


if __name__ == '__main__':
    main()
