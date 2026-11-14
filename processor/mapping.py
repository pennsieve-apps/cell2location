"""
Stage 2: mapping the signatures onto the spatial samples.

Every sample's counts (layers["counts"] from st-qc, else X from st-prep) are concatenated
(batch_key = sample_id: one detection efficiency per sample), the genes are intersected with
the signatures, and cell2location's spatial model is trained on all locations at once
(batch_size=None, train_size=1), as in the published tutorial. The posterior is summarised as
the 5 % quantile (q05, the conservative abundance the authors recommend) and the mean.
Mitochondrially encoded genes are left out of the mapping, as in the tutorial (their counts
reflect technical rather than cell-type differences).
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
from processor.reference import is_integer_valued

log = logging.getLogger("cell2location-app")

OBS_KEEP = ("x", "y", "in_tissue", "array_row", "array_col", "total_counts", "annotation", "condition")


def load_sample(path, sample_id: str) -> tuple[ad.AnnData, dict | None]:
    """A slim copy with integer counts in X, the contract obs columns and obsm["spatial"]; plus the
    lowres image and scale factors for plotting when the file has them."""
    adata = ad.read_h5ad(path)
    if "counts" in adata.layers:
        X, source = adata.layers["counts"], "layers.counts"
    else:
        X, source = adata.X, "X"
    if not is_integer_valued(X):
        raise ValueError(f"{source} is not integer counts (st-qc writes counts to layers['counts'])")
    X = sp.csr_matrix(X, dtype=np.float32) if not sp.isspmatrix_csr(X) else X.astype(np.float32)
    obs = pd.DataFrame(index=adata.obs_names.astype(str))
    obs["sample_id"] = adata.obs["sample_id"].astype(str).values if "sample_id" in adata.obs else sample_id
    for c in OBS_KEEP:
        if c in adata.obs:
            obs[c] = adata.obs[c].values
    var = pd.DataFrame(index=pd.Index(adata.var_names.astype(str)))
    for c in G.ID_COLUMNS + G.SYMBOL_COLUMNS:
        if c in adata.var:
            var[c] = adata.var[c].astype(str).values
    symbols, _ = G.symbols_and_ids(adata.var)
    mt = symbols.str.upper().str.startswith("MT-").to_numpy()
    if "mt" in adata.var:
        mt = mt | adata.var["mt"].fillna(False).astype(bool).to_numpy()
    var["mt"] = mt
    slim = ad.AnnData(X=X, obs=obs, var=var)
    if "spatial" in adata.obsm:
        slim.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
    elif {"x", "y"} <= set(adata.obs.columns):
        slim.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype=np.float64)
    else:
        raise ValueError("no obsm['spatial'] and no obs x/y columns")
    slim.var_names_make_unique()
    slim.uns["counts_source"] = source
    image = None
    lib = (adata.uns.get("spatial") or {}).get(sample_id) if "spatial" in adata.uns else None
    if lib is None and "spatial" in adata.uns and len(adata.uns["spatial"]) == 1:
        lib = next(iter(adata.uns["spatial"].values()))
    if isinstance(lib, dict):
        imgs, sf = lib.get("images", {}), lib.get("scalefactors", {})
        for key in ("lowres", "hires"):
            if key in imgs and f"tissue_{key}_scalef" in sf:
                image = {"image": np.asarray(imgs[key]), "scalef": float(sf[f"tissue_{key}_scalef"]),
                         "spot_diameter_fullres": float(sf.get("spot_diameter_fullres", 0) or 0), "key": key}
                break
    pennsieve = adata.uns.get("pennsieve", {}) if "pennsieve" in adata.uns else {}
    slim.uns["pennsieve"] = {k: pennsieve[k] for k in ("platform", "kind", "coordinate_unit", "spot_diameter", "grid")
                             if k in pennsieve}
    return slim, image


def concat_samples(adatas: list[ad.AnnData]) -> ad.AnnData:
    if len(adatas) == 1:
        out = adatas[0]
    else:
        n_genes = {a.n_vars for a in adatas}
        out = ad.concat(adatas, join="inner", merge="first", uns_merge=None, index_unique=None)
        if len(n_genes) > 1 or out.n_vars != adatas[0].n_vars:
            log.warning("samples have different genes (%s); using the %d shared by all", sorted(n_genes), out.n_vars)
    out.obs["sample_id"] = pd.Categorical(out.obs["sample_id"].astype(str))
    if not out.obs_names.is_unique:
        raise ValueError("obs_names are not unique across samples (the contract wants <sample_id>:<barcode>)")
    return out


def count_shared_genes(adata_vis: ad.AnnData, signatures: pd.DataFrame) -> int:
    """How many genes intersect() will keep (mitochondrial genes left out), without copying any data.
    signatures: index = symbol, optional gene_id column (the values are not used)."""
    var = adata_vis.var
    if "mt" in var:
        var = var[~var["mt"].to_numpy()]
    return len(G.match_genes(var, signatures))


def intersect(adata_vis: ad.AnnData, signatures: pd.DataFrame) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame, dict]:
    """Subset the spatial data and the signatures to the matched genes (same order).

    Returns (adata_vis subset, cell_state_df indexed by the spatial var_names, genes_used, counts)."""
    n_mt = int(adata_vis.var["mt"].sum()) if "mt" in adata_vis.var else 0
    if n_mt:
        adata_vis = adata_vis[:, ~adata_vis.var["mt"].to_numpy()]
    used = G.match_genes(adata_vis.var, signatures)
    counts = G.match_counts(used, len(signatures), adata_vis.n_vars)
    counts["n_mt_genes_excluded"] = n_mt
    if len(used) == 0:
        raise ValueError("no gene of the signatures matches the spatial data (symbols, case-insensitive symbols or "
                         "Ensembl ids)")
    types = G.signature_types(signatures)
    cell_state = pd.DataFrame(signatures.loc[used["gene"].values, types].values, index=used["spatial_gene"].values,
                              columns=types)
    sub = adata_vis[:, used["spatial_gene"].values].copy()
    assert list(sub.var_names) == list(cell_state.index)
    log.info("genes: %d of %d signature genes matched to the %d spatial genes (%d by Ensembl id, %d by symbol, "
             "%d by symbol ignoring case; %d mitochondrial genes left out)", counts["n_matched"], counts["n_signature_genes"], counts["n_spatial_genes"],
             counts["by_ensembl_id"], counts["by_symbol"], counts["by_symbol_case_insensitive"], n_mt)
    return sub, cell_state, used, counts


def train_mapping(adata_vis: ad.AnnData, cell_state: pd.DataFrame, n_cells_per_location: float, detection_alpha: float,
                  epochs: int, accelerator: str, seed: int) -> dict:
    """Cell2location -> {adata (with obsm q05/means), history, qc, seconds, final_elbo}."""
    from cell2location.models import Cell2location

    T.seed_everything(seed)
    Cell2location.setup_anndata(adata=adata_vis, batch_key="sample_id")
    mod = Cell2location(adata_vis, cell_state_df=cell_state, N_cells_per_location=float(n_cells_per_location),
                        detection_alpha=float(detection_alpha))
    t0 = time.monotonic()
    mod.train(batch_size=None, train_size=1, **T.train_kwargs("spatial", epochs, accelerator))
    seconds_train = time.monotonic() - t0
    adata_vis = mod.export_posterior(adata_vis, sample_kwargs={"num_samples": 1000, "batch_size": min(adata_vis.n_obs, 10000),
                                                               "accelerator": accelerator, "show_progress": False})
    seconds = time.monotonic() - t0
    qc = T.reconstruction_qc(mod, n_obs=1000, seed=seed)
    final = float(mod.history["elbo_train"].iloc[-1, 0])
    log.info("spatial mapping: %d locations x %d genes, %d cell types, %d samples, final ELBO %.6g, %.0fs (train %.0fs); "
             "reconstruction log1p-correlation %.3f", adata_vis.n_obs, adata_vis.n_vars, len(cell_state.columns),
             adata_vis.obs["sample_id"].nunique(), final, seconds, seconds_train, qc["log1p_correlation"])
    return {"adata": adata_vis, "history": T.history_table(mod), "qc": qc, "seconds": round(seconds, 1),
            "seconds_train": round(seconds_train, 1), "final_elbo": final}


def abundance_tables(adata_vis: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(q05, means): obs x cell type DataFrames with the plain cell type names as columns."""
    types = list(adata_vis.uns["mod"]["factor_names"])

    def table(summary: str) -> pd.DataFrame:
        key = f"{summary}_cell_abundance_w_sf"
        if key in adata_vis.obsm:
            df = adata_vis.obsm[key][[f"{summary}cell_abundance_w_sf_{t}" for t in types]].copy()
        else:
            df = adata_vis.obs[[f"{summary}cell_abundance_w_sf_{t}" for t in types]].copy()
        df.columns = types
        return df.astype(np.float64)

    return table("q05"), table("means")


def dominant_types(q05: pd.DataFrame, min_fraction: float) -> tuple[pd.Series, pd.Series]:
    """Per observation: the most abundant type, 'Mixed' when it holds < min_fraction of the total; and that fraction."""
    total = q05.sum(axis=1)
    top = q05.idxmax(axis=1).astype(str)
    frac = (q05.max(axis=1) / total.replace(0, np.nan)).fillna(0.0)
    label = top.where((frac >= min_fraction) & (total > 0), "Mixed")
    return label.rename("dominant_type"), frac.rename("dominant_fraction")
