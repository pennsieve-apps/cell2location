"""
Stage 3: co-location groups, cell2location's downstream NMF (CoLocatedGroupsSklearnNMF, the model
cell2location.run_colocation fits) on the q05 abundances: each group is a set of cell types that
tend to share locations (a tissue zone or micro-environment), with a weight per cell type (the
fraction of that type's abundance explained by the group) and a loading per location.

The model class is called directly (same defaults as run_colocation: random init, l1_ratio 0.5,
20000 iterations, 5 restarts) so the step does not write run_colocation's diagnostic plots and
pickles. One correction: run_colocation's regularisation alpha=0.01 was written for scikit-learn's
old, unscaled `alpha`; scikit-learn >= 1.2 multiplies alpha_W by the number of cell types and
alpha_H by the number of locations, which with thousands of spots zeroes most groups. The same
penalty is passed as alpha_W = 0.01 / n_cell_types and alpha_H = 0.01 / n_locations.

N_COLOCATION_GROUPS=0 tries 4..8 groups (fewer than the number of cell types) and keeps the
largest number whose groups are reproducible: NMF is refitted on BOOTSTRAPS seeded 80 %
subsamples of the locations, each group's cell-type loadings are matched to the first fit's
(Hungarian assignment on correlations, as cell2location's align_plot_stability does), and a
number of groups counts as stable when the worst-matched group still correlates >= STABLE on
average. When none is stable the most stable one is used.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

log = logging.getLogger("cell2location-app")

CANDIDATES = (4, 5, 6, 7, 8)
STABLE = 0.9
N_RESTARTS = 5
BOOTSTRAPS = 5
SUBSAMPLE = 0.8
ALPHA = 0.01
N_ITER = 20000


def matched_correlations(fac1: np.ndarray, fac2: np.ndarray) -> np.ndarray:
    """Correlation of each column of fac1 with its best-matched column of fac2 (one-to-one)."""
    k = fac1.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(fac1, fac2, rowvar=False)[:k, k:]
    corr = np.nan_to_num(corr, nan=-1.0)
    rows, cols = linear_sum_assignment(-corr)
    return corr[rows, cols]


def aligned_correlation(fac1: np.ndarray, fac2: np.ndarray) -> float:
    """Mean matched correlation (what cell2location's align_plot_stability shows)."""
    return float(np.mean(matched_correlations(fac1, fac2)))


def _nmf(X: np.ndarray, k: int, seed: int):
    from sklearn.decomposition import NMF

    m = NMF(n_components=k, init="random", random_state=seed, l1_ratio=0.5, alpha_W=ALPHA / X.shape[1],
            alpha_H=ALPHA / X.shape[0], max_iter=N_ITER, tol=1e-6)
    m.fit(X)
    return m


def stability(X: np.ndarray, k: int, seed: int, bootstraps: int = BOOTSTRAPS) -> dict:
    """Bootstrap stability of k groups: mean over refits of the worst group's matched correlation."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    fits = []
    for b in range(bootstraps + 1):
        idx = np.sort(rng.choice(n, max(k + 1, int(SUBSAMPLE * n)), replace=False))
        fits.append(_nmf(X[idx], k, seed + b))
    ref = fits[0].components_.T
    per = [matched_correlations(ref, f.components_.T) for f in fits[1:]]
    return {"n_groups": int(k), "stability": round(float(np.mean([p.min() for p in per])), 4),
            "mean_group_correlation": round(float(np.mean([p.mean() for p in per])), 4),
            "reconstruction_error": round(float(np.mean([f.reconstruction_err_ for f in fits])), 4)}


def fit(adata_vis, n_fact: int, seed: int, n_restarts: int = N_RESTARTS):
    """cell2location's CoLocatedGroupsSklearnNMF on all locations (scale-corrected alpha, see above)."""
    from cell2location.models.downstream import CoLocatedGroupsSklearnNMF

    X = np.asarray(adata_vis.obsm["q05_cell_abundance_w_sf"].values, dtype=np.float64)
    mod = CoLocatedGroupsSklearnNMF(
        int(n_fact), X, n_iter=N_ITER, verbose=False, var_names=list(adata_vis.uns["mod"]["factor_names"]),
        obs_names=adata_vis.obs_names, fact_names=[f"fact_{i}" for i in range(int(n_fact))],
        sample_id=adata_vis.obs["sample_id"], init="random", random_state=int(seed), alpha=ALPHA / X.shape[1],
        nmf_kwd_args={"tol": 1e-6, "alpha_H": ALPHA / X.shape[0]})
    mod.fit(n=int(n_restarts), n_type="restart")
    mod.sample2df(node_name="nUMI_factors", ct_node_name="cell_type_factors")
    return mod


def run(adata_vis, n_groups: int, seed: int, n_restarts: int = N_RESTARTS) -> dict:
    """{n_groups, reason, candidates: [{n_groups, stability, mean_group_correlation, reconstruction_error}],
    colocation (group, cell_type, weight), by_obs (obs x group loadings), group_summary}."""
    n_types = len(adata_vis.uns["mod"]["factor_names"])
    X = np.asarray(adata_vis.obsm["q05_cell_abundance_w_sf"].values, dtype=np.float64)
    if n_groups > 0:
        ks = [int(n_groups)]
    else:
        ks = [k for k in CANDIDATES if k < n_types] or ([max(2, n_types - 1)] if n_types > 2 else [n_types])
    candidates = [stability(X, k, seed) for k in ks]
    if n_groups > 0:
        chosen, reason = ks[0], f"N_COLOCATION_GROUPS={n_groups}"
    else:
        stable = [c for c in candidates if c["stability"] >= STABLE]
        if stable:
            chosen = max(c["n_groups"] for c in stable)
            reason = (f"largest number of groups in {ks[0]}-{ks[-1]} that is reproducible on {BOOTSTRAPS} seeded "
                      f"{int(SUBSAMPLE * 100)} % subsamples of the locations (every group's cell-type loadings "
                      f"correlate >= {STABLE} with the matched group, on average)")
        else:
            chosen = max(candidates, key=lambda c: c["stability"])["n_groups"]
            reason = f"no number of groups reached stability {STABLE}; the most stable one"
    log.info("co-location: %s; chosen %d groups (%s)",
             ", ".join(f"{c['n_groups']} groups: stability {c['stability']:.3f}" for c in candidates), chosen, reason)

    mod = fit(adata_vis, chosen, seed, n_restarts)
    loc = mod.location_factors_df.copy()                 # obs x factor, nUMI loadings
    frac = mod.cell_type_fractions.copy()                # cell type x factor
    loc.columns = [c.replace("mean_nUMI_factors", "") for c in loc.columns]
    frac.columns = [c.replace("mean_cell_type_factors", "") for c in frac.columns]
    # name the groups G01.. by decreasing total loading over locations
    order = np.argsort(-loc.values.sum(0))
    names = {loc.columns[j]: f"G{r + 1:02d}" for r, j in enumerate(order)}
    loc = loc.rename(columns=names)[[names[c] for c in loc.columns[order]]]
    frac.columns = [names[c] for c in frac.columns]
    frac = frac[list(loc.columns)].fillna(0.0)
    rows = [{"group": g, "cell_type": str(ct), "weight": float(frac.loc[ct, g])}
            for g in frac.columns for ct in frac.index]
    colocation = pd.DataFrame(rows, columns=["group", "cell_type", "weight"])
    summary = []
    for g in frac.columns:
        w = frac[g].sort_values(ascending=False)
        summary.append({"group": g, "top_types": [f"{ct} ({v:.2f})" for ct, v in w.head(4).items()],
                        "mean_loading": round(float(loc[g].mean()), 4)})
    dead = [g for g in loc.columns if loc[g].sum() == 0]
    if dead:
        log.warning("co-location: groups %s are empty", ", ".join(dead))
    return {"n_groups": int(chosen), "reason": reason, "candidates": candidates, "colocation": colocation,
            "by_obs": loc.loc[adata_vis.obs_names], "group_summary": summary, "n_restarts": int(n_restarts)}
