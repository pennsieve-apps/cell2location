"""
Stage 1: reference cell-type signatures from a scRNA-seq .h5ad.

Raw counts come from layers["counts"], X or .raw (the first that is integer-valued); the
cell type from CELL_TYPE_COLUMN; an optional REFERENCE_BATCH_COLUMN is the batch_key of the
regression model (donor, sample or library). Cell types with fewer than MIN_CELLS_PER_TYPE
cells are dropped, the rest subsampled to MAX_CELLS_PER_TYPE (seeded); genes are filtered with
cell2location's filter_genes; RegressionModel (the negative-binomial regression of
cell2location) is trained REFERENCE_EPOCHS epochs and the posterior mean per cell type and gene
(means_per_cluster_mu_fg) is the signature.
"""
from __future__ import annotations

import logging
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from processor import genes as G
from processor import training as T

log = logging.getLogger("cell2location-app")


def is_integer_valued(X, n_check: int = 200_000) -> bool:
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if data.size == 0:
        return False
    data = data[:n_check]
    return bool(np.all(np.isfinite(data)) and np.allclose(data, np.round(data)) and data.min() >= 0)


def counts_of(adata: ad.AnnData) -> tuple[sp.csr_matrix, pd.DataFrame, str]:
    """(integer counts as CSR float32, the matching var table, where they came from)."""
    candidates = []
    if "counts" in adata.layers:
        candidates.append(("layers.counts", adata.layers["counts"], adata.var))
    candidates.append(("X", adata.X, adata.var))
    if adata.raw is not None:
        candidates.append(("raw.X", adata.raw.X, adata.raw.var))
    for name, X, var in candidates:
        if X is None:
            continue
        if is_integer_valued(X):
            M = sp.csr_matrix(X, dtype=np.float32) if not sp.isspmatrix_csr(X) else X.astype(np.float32)
            return M, var.copy(), name
    raise ValueError("no integer count matrix in layers['counts'], X or .raw.X (cell2location needs raw UMI counts)")


def load_reference(path, cell_type_col: str, batch_col: str) -> ad.AnnData:
    adata = ad.read_h5ad(path)
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"reference has no obs column {cell_type_col!r} (CELL_TYPE_COLUMN); columns: "
                         f"{', '.join(map(str, adata.obs.columns[:20]))}")
    if batch_col and batch_col not in adata.obs.columns:
        raise ValueError(f"reference has no obs column {batch_col!r} (REFERENCE_BATCH_COLUMN)")
    X, var, source = counts_of(adata)
    obs = pd.DataFrame(index=adata.obs_names.astype(str))
    obs["cell_type"] = adata.obs[cell_type_col].astype(str).values
    if batch_col:
        obs["batch"] = adata.obs[batch_col].astype(str).values
    symbols, ids = G.symbols_and_ids(var)
    new_var = pd.DataFrame(index=pd.Index(var.index.astype(str)))
    new_var["symbol"] = symbols.values
    new_var["gene_id"] = ids.values if ids is not None else ""
    ref = ad.AnnData(X=X, obs=obs, var=new_var)
    ref.var_names_make_unique()
    ref.uns["counts_source"] = source
    log.info("reference: %d cells x %d genes, counts from %s, %d cell types%s", ref.n_obs, ref.n_vars, source,
             ref.obs["cell_type"].nunique(), f", batch = {batch_col}" if batch_col else "")
    return ref


def select_cells(ref: ad.AnnData, min_cells: int, max_cells: int, seed: int) -> tuple[ad.AnnData, pd.DataFrame]:
    """Drop cell types with < min_cells cells, subsample the rest to max_cells (seeded); the summary table."""
    rng = np.random.default_rng(seed)
    counts = ref.obs["cell_type"].value_counts()
    keep_idx, rows = [], []
    for ct in sorted(counts.index):
        idx = np.flatnonzero((ref.obs["cell_type"] == ct).values)
        n = len(idx)
        if n < min_cells:
            rows.append({"cell_type": ct, "n_cells": n, "n_used": 0, "status": f"dropped: fewer than {min_cells} cells"})
            continue
        if max_cells > 0 and n > max_cells:
            idx = np.sort(rng.choice(idx, max_cells, replace=False))
        keep_idx.append(idx)
        rows.append({"cell_type": ct, "n_cells": n, "n_used": len(idx), "status": "used"})
    summary = pd.DataFrame(rows, columns=["cell_type", "n_cells", "n_used", "status"])
    if not keep_idx:
        raise ValueError(f"no cell type has at least {min_cells} cells")
    keep = np.sort(np.concatenate(keep_idx))
    out = ref[keep].copy()
    out.obs["cell_type"] = pd.Categorical(out.obs["cell_type"].astype(str))
    dropped = summary[summary["status"] != "used"]
    log.info("reference cells: %d of %d used, %d cell types (%d dropped%s)", out.n_obs, ref.n_obs,
             (summary["status"] == "used").sum(), len(dropped),
             f": {', '.join(dropped['cell_type'])}" if len(dropped) else "")
    return out, summary


def filter_genes(ref: ad.AnnData, cell_count_cutoff: float, cell_percentage_cutoff2: float,
                 nonz_mean_cutoff: float, figure_path=None) -> ad.AnnData:
    """cell2location.utils.filtering.filter_genes; its diagnostic plot is saved when a path is given."""
    import matplotlib.pyplot as plt
    from cell2location.utils.filtering import filter_genes as c2l_filter

    n_cells = np.asarray((ref.X > 0).sum(0)).ravel()
    ref = ref[:, n_cells > 0].copy()      # genes never detected have no non-zero mean (log of 0)
    plt.close("all")
    selected = c2l_filter(ref, cell_count_cutoff=cell_count_cutoff, cell_percentage_cutoff2=cell_percentage_cutoff2,
                          nonz_mean_cutoff=nonz_mean_cutoff)
    if figure_path is not None:
        try:
            plt.gcf().savefig(figure_path, dpi=150)
        except Exception as exc:   # the plot is a convenience
            log.warning("gene filter plot failed: %s", exc)
    plt.close("all")
    out = ref[:, selected].copy()
    log.info("reference genes: %d of %d kept (cell_count_cutoff %g, cell_percentage_cutoff2 %g, nonz_mean_cutoff %g)",
             out.n_vars, ref.n_vars, cell_count_cutoff, cell_percentage_cutoff2, nonz_mean_cutoff)
    return out


def train_signatures(ref: ad.AnnData, batch: bool, epochs: int, accelerator: str, seed: int) -> dict:
    """RegressionModel -> {signatures (gene x cell type, + gene_id), history, qc, cluster_means, seconds}."""
    from cell2location.models import RegressionModel

    T.seed_everything(seed)
    ref.obs["_batch"] = ref.obs["batch"].astype(str).values if batch else "all"
    RegressionModel.setup_anndata(adata=ref, batch_key="_batch", labels_key="cell_type")
    mod = RegressionModel(ref)
    t0 = time.monotonic()
    mod.train(batch_size=2500, train_size=1, lr=0.002, **T.train_kwargs("reference", epochs, accelerator))
    seconds_train = time.monotonic() - t0
    ref = mod.export_posterior(ref, sample_kwargs={"num_samples": 1000, "batch_size": 2500,
                                                   "accelerator": accelerator, "show_progress": False})
    seconds = time.monotonic() - t0
    types = list(ref.uns["mod"]["factor_names"])
    key = "means_per_cluster_mu_fg"
    if key in ref.varm:
        est = ref.varm[key][[f"{key}_{t}" for t in types]].copy()
    else:
        est = ref.var[[f"{key}_{t}" for t in types]].copy()
    est.columns = types
    sig = pd.DataFrame(est.values, index=G.make_unique(ref.var["symbol"]), columns=types)
    sig.insert(0, "gene_id", ref.var["gene_id"].astype(str).values)
    if not (sig["gene_id"].str.len() > 0).any():
        sig = sig.drop(columns="gene_id")

    qc = T.reconstruction_qc(mod, n_obs=1000, seed=seed)
    # Estimated signatures vs the plain per-cluster mean of the counts (the second plot of plot_QC).
    aver = mod._compute_cluster_averages()[types]
    qc["cluster_means"] = aver.values.ravel()
    qc["estimated"] = est.loc[aver.index].values.ravel()
    log.info("reference signatures: %d genes x %d cell types, final ELBO %.5g, %.0fs (train %.0fs); reconstruction "
             "log1p-correlation %.3f", sig.shape[0], len(types), float(mod.history["elbo_train"].iloc[-1, 0]),
             seconds, seconds_train, qc["log1p_correlation"])
    return {"signatures": sig, "history": T.history_table(mod), "qc": qc, "seconds": round(seconds, 1),
            "seconds_train": round(seconds_train, 1), "final_elbo": float(mod.history["elbo_train"].iloc[-1, 0]),
            "n_batches": int(ref.obs["_batch"].nunique())}
