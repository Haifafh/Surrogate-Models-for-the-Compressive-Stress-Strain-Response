# Surrogate models for compressive stress–strain response of nanosilica/PVA ECC

Code, digitised data and results for the manuscript:

> **Surrogate Models for the Compressive Stress–Strain Response of Nanosilica/PVA-Modified
> Engineered Cementitious Composites: A Leakage-Free, Blind-Mixture Benchmark of
> Machine-Learning, Physics-Informed, Spline, and Operator Architectures**
> H. F. Alhasson, M. A. Albadrani, S. S. Alharbi — *Polymers* (MDPI), 2026.

H. F. Alhasson, M. Albadrani, S. S. Alharbi, _Polymers (submitted).

## What this repository is for

The paper's central claim is **methodological, not architectural**: when a stress–strain
curve dataset is split *point by point*, neighbouring strain increments of the same
specimen land on both sides of the split, so the reported R² measures interpolation along
the strain axis rather than generalisation to a new mix design. This repository exists so
that the blind-mixture protocol, the 21 surrogate formulations and every number in the
paper can be reproduced and reused.

**Headline result.** Under the blind protocol the physics-informed perceptron that leads
the ranking carries **no active physics penalty** — its weights were auto-selected to zero,
making it numerically identical to its data-only control. Accuracy and physical
admissibility are independent attributes: no data-driven surrogate in this benchmark
achieves damage admissibility, and across fifteen resampled hold-outs only a
gradient-boosted tree separates itself (2.77 ± 0.61 MPa, *p* = 0.002).

## Quick start

```bash
pip install -r requirements.txt

python scripts/01_build_dataset.py      # rebuild the 737-point dataset from the workbook
python scripts/02_smoke_benchmark.py    # fast end-to-end check, no PINN training
python src/tangent_analysis.py          # regenerate Table 17 from stored predictions
```

`02_smoke_benchmark.py` reproduces the Linear and Random-Forest rows of the blind table in
under a minute and is the recommended first check that your environment is sound.

## Data

| File | Contents |
|---|---|
| `data/dataset_curves.csv` | The 737 collocation points: `nS`, `PVA`, `strain_um_m`, `strain_pct`, `eps_frac`, `stress_MPa` |
| `data/peak_properties.csv` | Peak stress, peak strain, E₀, work-to-peak and ductility index per curve |
| `data/peak_properties_companion.csv` | Peak values published in the companion study, used for the provenance analysis |
| `data/raw/curves_source_workbook.xlsx` | Original digitisation workbook, sheets `a`–`f` = nS 0–5 %, column pairs N1–N4 = PVA 0.5/1.0/1.5/2.0 % |

Twenty-four curves span nS = 0–5 % × PVA = 0.5–2.0 %. Strains are digitised on a
**100 µε** grid; `strain_um_m` is microstrain, `strain_pct` = microstrain × 1e-4, and
`eps_frac` = microstrain × 1e-6.

### Provenance caveat — please read before using the absolute numbers

The digitised peaks differ from Table 5 of the companion study by a **mean of 13.0 %**
(maximum **27.3 %** at 2 % nS / 0.5 % PVA), and that source is internally inconsistent
(its abstract quotes 80 MPa for a mixture whose table gives 63 MPa). Every error statistic
in the paper therefore inherits this floor. **Absolute RMSE values are relative rankings,
not calibrated design accuracies.** `results/table5_sensitivity_summary.csv` and
`results/table5_rank_corr.json` quantify how far the ranking moves when the published
peaks are substituted for the digitised ones (Spearman ρ = 0.43, *p* = 0.40).

## Repository layout

```
data/                 digitised curves, peak properties, raw workbook
src/                  model code, experiment drivers, figure and table generators
  pipeline.py           dataset construction, splits, modulus surface, PINN trainer
  pinn.py               physics-informed perceptron (L_data + lambda_E L_E + lambda_D L_D + lambda_S L_S)
  kan.py                spline / Kolmogorov-Arnold backbone
  deeponet.py           branch-trunk operator backbone
  train_compare.py      cross-architecture comparison under a matched objective
  run2.py               PIKAN + DeepONet experiment driver
  predict_all.py        refit on the blind training set, predict all 24 compositions
  analysis2.py          seed variance, significance tests
  analysis3.py          resampled-split and multi-network analyses
  metrics_ext.py        extended accuracy metrics (Table 7)
  tangent_analysis.py   tangent-modulus diagnostics (Table 17)
  figures.py            all result figures -> native TikZ/PGF
  mkfigs.py             architecture and operator schematics -> native TikZ/PGF
  tables2.py            LaTeX table generation
scripts/
  01_build_dataset.py   workbook -> dataset_curves.csv (reproduces to 1e-14 MPa)
  02_smoke_benchmark.py fast verification
results/              all CSV/JSON/npz outputs quoted in the manuscript
```

## Reproducing specific results

| Manuscript item | Command | Output |
|---|---|---|
| Table 6, 9 (blind benchmark, diagnostics) | `python src/predict_all.py` then `src/analysis2.py` | `results/benchmark_metrics.csv`, `results/physical_diagnostics.csv` |
| Table 7 (extended metrics) | `python src/metrics_ext.py` | `results/metrics_extended.csv` |
| Table 10 (physics-loss ablation) | `python src/pipeline.py` | `results/ablation.csv` |
| Tables 11, 12 (cross-architecture, multi-network) | `python src/train_compare.py`, `python src/run2.py` | `results/blind_results.csv`, `results/multisplit_new_summary.csv` |
| Table 14 (parameter prediction, LOOCV) | `python src/analysis3.py` | `results/peak_loocv.csv` |
| Tables 15, 16 (significance) | as above | `results/significance.json`, `results/significance_new.json` |
| **Table 17 (tangent diagnostics)** | `python src/tangent_analysis.py` | `results/table17_tangent.csv` |


The stored `.npz` archives (`results/preds2.npz`, `results/pred_all.npz`) contain the
blind-mixture predictions, so Tables 7 and 17 can be regenerated **without retraining**.

## Models compared (21 formulations)

17 baselines — linear, ridge, polynomial (deg 3), k-NN, SVR, decision tree, random forest,
extra trees, gradient boosting, histogram gradient boosting, Gaussian process, shallow and
deep MLPs, the closed-form Carreira–Chu law and a hybrid parameter-transfer model — plus
4 physics-informed formulations: a perceptron PINN, a spline (Kolmogorov–Arnold) network,
a DeepONet, and its physics-informed variant.

Three physics residuals are available: an **elastic-modulus pin** (initial tangent equal to
a fitted modulus surface), a **damage-consistency** term (D = 1 − σ/(E₀ε) non-decreasing)
and a **curvature-smoothness** term.

## Evaluation protocols

* **Split A (interpolating)** — random 80/20 over collocation points, 589/148.
* **Split B (blind mixture)** — five complete compositions withheld: (1.0, 0.5),
  (2.0, 1.5), (3.0, 2.0), (4.0, 1.0), (5.0, 1.5) % nS/% PVA, 578/159 points.

Split B is the operationally relevant one. Split A is retained only to show how much it
inflates R².

## Licence and citation

Code MIT (see `LICENSE`); data CC BY 4.0. Citation metadata in `CITATION.cff`.
If you use the blind-mixture protocol or the dataset, please cite the paper above.
