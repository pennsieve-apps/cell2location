"""
Stage 3: co-location groups, cell2location's downstream NMF (run_colocation with
CoLocatedGroupsSklearnNMF) on the q05 abundances: each group is a set of cell types that tend
to share locations (a tissue zone or micro-environment), with a weight per cell type (the
fraction of that type's abundance explained by the group) and a loading per location.

This is what cell2location.run_colocation fits (same model and defaults); the model class is
called directly so the step does not write run_colocation's diagnostic plots and pickles.

N_COLOCATION_GROUPS=0 tries 4..8 groups (fewer than the number of cell types) and keeps the largest number that is stable across
the NMF's own random restarts (the diagnostic cell2location plots with evaluate_stability:
the correlation of the aligned cell-type loadings between restarts; stable = mean >= 0.9), or
the most stable one when none reaches that.
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


def aligned_correlation(fac1: np.ndarray, fac2: np.ndarray) -> float:
    """Mean correlation of the columns of fac1 with the best-matched columns of fac2
    (what cell2location's align_plot_stability shows)."""
    k = fac1.shape[1]
    corr = np.corrcoef(fac1, fac2, rowvar=False)[:k, k:]
    corr = np.nan_to_num(corr, nan=-1.0)
    perm = linear_sum_assignment(2 - corr)[1]
    return float(np.mean(corr[np.arange(k), perm]))


def stability(mod) -> float:
    """Mean aligned correlation of cell_type_factors between restart 1 and every other restart."""
    inits = sorted(mod.results)
    ref = mod.results[inits[0]]["post_sample_means"]["cell_type_factors"]
    scores = [aligned_correlation(ref, mod.results[i]["post_sample_means"]["cell_type_factors"]) for i in inits[1:]]
    return float(np.mean(scores)) if scores else float("nan")


def fit(adata_vis, n_fact: int, seed: int, n_restarts: int):
    """cell2location's CoLocatedGroupsSklearnNMF with run_colocation's defaults (init random, alpha 0.01,
    l1_ratio 0.5, tol 1e-6, 20000 iterations, n_restarts restarts) on the q05 abundances, without run_colocation's
    plots and files."""
    from cell2location.models.downstream import CoLocatedGroupsSklearnNMF

    X = np.asarray(adata_vis.obsm["q05_cell_abundance_w_sf"].values, dtype=np.float64)
    mod = CoLocatedGroupsSklearnNMF(
        int(n_fact), X, n_iter=20000, verbose=False, var_names=list(adata_vis.uns["mod"]["factor_names"]),
        obs_names=adata_vis.obs_names, fact_names=[f"fact_{i}" for i in range(int(n_fact))],
        sample_id=adata_vis.obs["sample_id"], init="random", random_state=int(seed), alpha=0.01,
        nmf_kwd_args={"tol": 1e-6})
    mod.fit(n=int(n_restarts), n_type="restart")
    mod.sample2df(node_name="nUMI_factors", ct_node_name="cell_type_factors")
    return mod


def run(adata_vis, n_groups: int, seed: int, n_restarts: int = N_RESTARTS) -> dict:
    """{n_groups, reason, candidates: [{n_groups, stability, reconstruction_error}], colocation (group, cell_type,
    weight), by_obs (obs x group loadings), group_summary}."""
    n_types = len(adata_vis.uns["mod"]["factor_names"])
    ks = [int(n_groups)] if n_groups > 0 else [k for k in CANDIDATES if k < n_types]
    if not ks:
        ks = [max(2, n_types - 1)] if n_types > 2 else [n_types]
    mods = {k: fit(adata_vis, k, seed, n_restarts) for k in ks}

    candidates = []
    for k in ks:
        mod = mods[k]
        err = float(np.mean([m.reconstruction_err_ for m in mod.models.values()]))
        candidates.append({"n_groups": k, "stability": round(stability(mod), 4), "reconstruction_error": round(err, 4)})
    if n_groups > 0:
        chosen, reason = ks[0], f"N_COLOCATION_GROUPS={n_groups}"
    else:
        stable = [c for c in candidates if c["stability"] >= STABLE]
        if stable:
            chosen = max(c["n_groups"] for c in stable)
            reason = (f"largest number of groups whose cell-type loadings are stable across {n_restarts} NMF restarts "
                      f"(mean aligned correlation >= {STABLE})")
        else:
            best = max(candidates, key=lambda c: c["stability"])
            chosen = best["n_groups"]
            reason = f"no candidate reached stability {STABLE}; the most stable one"
    log.info("co-location: %s; chosen %d groups (%s)",
             ", ".join(f"{c['n_groups']} groups: stability {c['stability']:.3f}" for c in candidates), chosen, reason)

    mod = mods[chosen]
    loc = mod.location_factors_df.copy()                 # obs x factor, nUMI loadings
    frac = mod.cell_type_fractions.copy()                # cell type x factor
    loc.columns = [c.replace("mean_nUMI_factors", "") for c in loc.columns]
    frac.columns = [c.replace("mean_cell_type_factors", "") for c in frac.columns]
    # name the groups G01.. by decreasing total loading over locations
    order = np.argsort(-loc.values.sum(0))
    names = {loc.columns[j]: f"G{r + 1:02d}" for r, j in enumerate(order)}
    loc = loc.rename(columns=names)[[names[c] for c in loc.columns[order]]]
    frac.columns = [names[c] for c in frac.columns]
    frac = frac[list(loc.columns)]
    rows = [{"group": g, "cell_type": str(ct), "weight": float(frac.loc[ct, g])}
            for g in frac.columns for ct in frac.index]
    colocation = pd.DataFrame(rows, columns=["group", "cell_type", "weight"])
    summary = []
    for g in frac.columns:
        w = frac[g].sort_values(ascending=False)
        summary.append({"group": g, "top_types": [f"{ct} ({v:.2f})" for ct, v in w.head(4).items()],
                        "mean_loading": round(float(loc[g].mean()), 4)})
    return {"n_groups": int(chosen), "reason": reason, "candidates": candidates, "colocation": colocation,
            "by_obs": loc.loc[adata_vis.obs_names], "group_summary": summary, "n_restarts": int(n_restarts)}
