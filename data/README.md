# Data

## Files

| File | Description |
|------|-------------|
| `raw/curves_source_workbook.xlsx` | Original digitised workbook. Sheets `a`–`f` = nanosilica content 0–5 % by mass of binder; column pairs `N1`–`N4` = PVA fibre volume fraction 0.5 / 1.0 / 1.5 / 2.0 %. Each pair is `(strain, stress)` with strain in units of 10³ µε and stress in MPa. |
| `dataset_curves.csv` | **The working dataset.** 737 collocation points: all 24 curves linearly interpolated onto a common 100 µε axial-strain grid. Columns: `nS`, `PVA`, `strain_um_m` (µε), `strain_pct`, `eps_frac`, `stress_MPa`. |
| `peak_properties.csv` | Per-mixture peak stress, peak strain, initial tangent modulus `E0` (least-squares slope below 0.4 σ_p), point count and maximum strain. |

## Design space

24 mixtures = 6 nanosilica contents (0, 1, 2, 3, 4, 5 % by mass of binder) × 4 PVA
volume fractions (0.5, 1.0, 1.5, 2.0 %). Peak strengths span 48.7–80.0 MPa.

## Blind-mixture split (Split B)

Five **complete** curves are withheld from training, so the test set measures
extrapolation in mix-design space rather than interpolation:

    (nS %, PVA %) = (1.0, 0.5), (2.0, 1.5), (3.0, 2.0), (4.0, 1.0), (5.0, 1.5)

This leaves 578 training points and 159 blind test points. `Split A` (a random
point-wise 80/20 partition) is also available for comparison, but it places points
from the same curve in both partitions and therefore overstates generalisation.

## Provenance caveat — please read before using these data

The digitised peak strengths agree with the characteristic values quoted in the
text and abstract of the companion experimental study (e.g. 80 MPa at 2 % nS /
0.5 % PVA), but they differ systematically from the values tabulated in Table 5
of that study: mean absolute deviation **13.0 %** in peak stress (maximum 27.3 %)
and 6.6 % in peak strain, with 0 of 20 mixtures agreeing to within 1 %.

The source document is internally inconsistent — for the 2 % nS / 0.5 % PVA
mixture its own abstract reports 80 MPa while its Table 5 lists 63 MPa. The
digitised curves were therefore treated as the authoritative material response,
and every model comparison in the accompanying manuscript is made against this
single consistent dataset. The discrepancy affects **absolute** error magnitudes
but not the relative ranking of architectures. See `docs/REPRODUCIBILITY.md`.

## Redistribution

The digitised data derive from the experimental programme of the companion study.
If you redistribute them, cite that study (see `CITATION.cff`).
