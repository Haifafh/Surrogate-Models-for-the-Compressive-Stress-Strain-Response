#!/usr/bin/env python3
"""Rebuild data/dataset_curves.csv and data/peak_properties.csv from the raw workbook."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import data as D

REPO = D.REPO_ROOT
raw = D.load_raw_curves()
df = D.resample(raw)
pr = D.peak_properties(raw, df)
df.to_csv(os.path.join(REPO, 'data', 'dataset_curves.csv'), index=False)
pr.to_csv(os.path.join(REPO, 'data', 'peak_properties.csv'), index=False)
print(f'wrote {len(df)} collocation points, {len(pr)} mixtures')
print(f'peak strength range: {pr.sigma_peak.min():.1f}-{pr.sigma_peak.max():.1f} MPa')
print(f'initial modulus range: {pr.E0_GPa.min():.1f}-{pr.E0_GPa.max():.1f} GPa')
