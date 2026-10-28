#!/usr/bin/env python3
"""Regenerate the selected multi-panel figures as separate lettered parts.

For each panel group the non-member axes are temporarily detached from the figure
before saving, so each part file contains ONLY its own panel's TikZ code (no
duplicated content) and can be cropped exactly to that panel's bounding box.
"""
import os, sys, traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

ROOT = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(ROOT, 'figures')
sys.path.insert(0, ROOT)

import figures as F
orig_save = F.save
L = 'abcdefghijklmnopqrstuvwxyz'

TARGETS = {
    'fig01_experimental_curves': (6,  [[i] for i in range(6)], 3, 0.32),
    'fig04_predicted_ns_curves': (18, [list(range(0,6)), list(range(6,12)), list(range(12,18))], 1, 0.98),
    'fig06_blind_curves':        (5,  [[i] for i in range(5)], 3, 0.32),
    'fig07_parity_blind':        (16, [list(range(0,4)), list(range(4,8)), list(range(8,12)), list(range(12,16))], 1, 0.98),
    'fig09_ablation':            (4,  [[i] for i in range(4)], 2, 0.48),
    'fig10_digital_twin':        (2,  [[0],[1]], 2, 0.48),
    'fig11_uncertainty':         (5,  [[i] for i in range(5)], 3, 0.32),
    'fig12_response_surface':    (4,  [[i] for i in range(4)], 2, 0.48),
    'p2fig04_smoothness':        (2,  [[0],[1]], 2, 0.48),
    'p3fig02_modes':             (2,  [[0],[1]], 2, 0.48),
    'fig13_reviewer_analytics':  (2,  [[0],[1]], 2, 0.48),
}

def save_parts(fig, name):
    if name not in TARGETS:
        return orig_save(fig, name)
    exp, groups, cols, w = TARGETS[name]
    all_axes = list(fig.axes)
    exp = min(exp, len(all_axes))
    groups = [list(g) for g in groups]
    # After the split every part is a STANDALONE subfigure, so a panel whose tick
    # labels were suppressed by the original grid's sharex/sharey must get them back
    # (otherwise it shows an axis label with no numbers).  Groups of >2 panels are
    # still one contiguous row, so their inner suppression stays.
    for g in groups:
        if len(g) <= 2:
            for i in g:
                try:
                    all_axes[i].tick_params(labelleft=True, labelbottom=True)
                except Exception:
                    pass
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tbs = []
    for ax in all_axes:
        bb = ax.get_tightbbox(rend)
        if bb is None:
            p = ax.get_position()
            bb = Bbox([[p.x0, p.y0], [p.x1, p.y1]])
        else:
            bb = bb.transformed(fig.dpi_scale_trans.inverted())
        tbs.append(bb)
    # auxiliary axes (colour bars) -> nearest group by horizontal adjacency
    for ei in range(exp, len(all_axes)):
        eb = tbs[ei]
        best, bd = 0, 1e9
        for gi, idxs in enumerate(groups):
            d = abs(eb.x0 - max(tbs[i].x1 for i in idxs))
            if d < bd:
                bd, best = d, gi
        groups[best].append(ei)
    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    for gi, idxs in enumerate(groups):
        keep = [all_axes[i] for i in idxs]
        drop = [a for a in all_axes if a not in keep]
        for a in drop:
            fig.delaxes(a)
        x0 = min(tbs[i].x0 for i in idxs); x1 = max(tbs[i].x1 for i in idxs)
        y0 = min(tbs[i].y0 for i in idxs); y1 = max(tbs[i].y1 for i in idxs)
        out = os.path.join(PARTS, '%s_part%s.pgf' % (name, L[gi]))
        fig.savefig(out, format='pgf', bbox_inches=Bbox([[x0, y0], [x1, y1]]))
        for a in drop:
            fig.add_axes(a)
        print('   -> %s_part%s.pgf' % (name, L[gi]))
    plt.close(fig)

F.save = save_parts
print('=== figures.py ==='); F.main()
print('=== mkfigs.py ===')
try:
    mp = '/home/user/workspace/paper2/mkfigs.py'
    exec(compile(open(mp, encoding='utf-8').read(), mp, 'exec'),
         {'__name__': '__main__', '__file__': mp})
except Exception:
    traceback.print_exc()
print('=== tables2.py ===')
try:
    tp = os.path.join(ROOT, 'tables2.py')
    exec(compile(open(tp, encoding='utf-8').read(), tp, 'exec'),
         {'__name__': '__main__', '__file__': tp})
except Exception:
    traceback.print_exc()
bad = []
for name, (exp, groups, cols, w) in TARGETS.items():
    for gi in range(len(groups)):
        f = os.path.join(PARTS, '%s_part%s.pgf' % (name, L[gi]))
        if not os.path.exists(f) or os.path.getsize(f) < 500:
            bad.append(os.path.basename(f))
print('MISSING/SMALL:', bad if bad else 'none')
