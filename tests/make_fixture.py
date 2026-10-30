"""
A tiny synthetic input for cell2location that follows the workflow contract, for the unit tests
and `make run`:

  python -m tests.make_fixture data/fixture [--layout linked|flat|both] [--seed 7]

- reference/reference.h5ad: a scRNA-seq reference of 5 cell types (B_cell, T_cell, Stromal,
  Myeloid at 120 cells each, and Rare with 6 cells, below MIN_CELLS_PER_TYPE), 160 genes with
  Ensembl-style var_names (ENSG...) and symbols in var["feature_name"], raw counts in X,
  obs["cell_type"] and obs["donor"] (two donors).
- spatial/<sample_id>.h5ad, two st-qc-style Visium samples (`lymph_a`, `lymph_b`) of 14 x 14 =
  196 hex-grid spots: X log-normalized, raw counts in layers["counts"], symbols as var_names and
  var["gene_ids"]. The counts are simulated from the same signatures with a planted structure:
  B cells concentrated in two round "follicles", T cells everywhere around them, Stromal cells
  evenly, Myeloid cells in one corner. `lymph_a` has a lowres "H&E" image and scalefactors.
  Six spatial genes are upper-case only in the spatial data (`GENE3` vs `Gene3` in the
  reference) so the case-insensitive symbol match is exercised when matching by symbol.
- metadata/samples.csv (sample_id, path, platform, kind, n_obs, ...).

PLANTED gives the truth the tests check against.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

TYPES = ["B_cell", "T_cell", "Stromal", "Myeloid"]
N_GENES = 160
MARKERS_PER_TYPE = 25
FOLLICLES = [(4.0, 4.0), (9.5, 9.0)]     # (row, col) in grid units
FOLLICLE_RADIUS = 2.6
GRID = 14
SAMPLES = ("lymph_a", "lymph_b")
PLANTED = {"types": TYPES, "dropped": "Rare", "follicle_type": "B_cell", "around_type": "T_cell"}


def symbols() -> list[str]:
    return [f"Gene{i}" for i in range(N_GENES)]


def signatures(rng: np.random.Generator) -> pd.DataFrame:
    """Expected counts per cell (genes x types): a low background plus 25 markers per type."""
    base = rng.gamma(0.6, 0.15, (N_GENES, len(TYPES) + 1))
    for t in range(len(TYPES) + 1):
        idx = slice(t * MARKERS_PER_TYPE, (t + 1) * MARKERS_PER_TYPE)
        base[idx, t] += rng.gamma(4.0, 0.8, MARKERS_PER_TYPE)
    return pd.DataFrame(base, index=symbols(), columns=TYPES + ["Rare"])


def nb(rng, mu, dispersion=0.2):
    shape = 1.0 / dispersion
    return rng.poisson(rng.gamma(shape, mu / shape)).astype(np.float32)


def make_reference(rng, sig: pd.DataFrame) -> ad.AnnData:
    blocks, types, donors = [], [], []
    for t in sig.columns:
        n = 6 if t == "Rare" else 120
        lib = rng.lognormal(0.0, 0.3, n)
        donor = rng.integers(0, 2, n)
        donor_effect = np.where(donor[:, None] == 1, 1.3, 1.0)
        blocks.append(nb(rng, np.outer(lib, sig[t].to_numpy()) * donor_effect))
        types += [t] * n
        donors += [f"donor{d + 1}" for d in donor]
    X = sp.csr_matrix(np.vstack(blocks))
    obs = pd.DataFrame({"cell_type": pd.Categorical(types), "donor": pd.Categorical(donors)},
                       index=[f"cell{i}" for i in range(X.shape[0])])
    var = pd.DataFrame({"feature_name": symbols()}, index=[f"ENSG{i:011d}" for i in range(N_GENES)])
    return ad.AnnData(X=X, obs=obs, var=var)


def abundance_grid(rng, sample_index: int):
    rows, cols = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    shift = 0.6 * sample_index
    d = np.min([np.hypot(rows - r - shift, cols - c) for r, c in FOLLICLES], axis=0)
    follicle = d < FOLLICLE_RADIUS
    w = np.zeros((rows.size, len(TYPES)))
    w[:, 0] = np.where(follicle, 6.0, 0.3)                      # B cells in follicles
    w[:, 1] = np.where(follicle, 0.8, 4.5)                      # T cells around them
    w[:, 2] = 1.5                                               # stroma everywhere
    w[:, 3] = np.where((rows > GRID - 5) & (cols < 5), 4.0, 0.2)  # myeloid corner
    w = rng.gamma(8.0, w / 8.0)
    return rows, cols, follicle, w


def make_spatial(rng, sig: pd.DataFrame, sid: str, sample_index: int, with_image: bool) -> ad.AnnData:
    rows, cols, follicle, w = abundance_grid(rng, sample_index)
    array_col = cols * 2 + (rows % 2)
    spacing = 200.0
    x = 1000.0 + array_col * spacing / 2.0
    y = 1000.0 + rows * spacing * 0.866
    detection = 0.6 if sample_index == 0 else 0.9
    mu = w @ sig[TYPES].to_numpy().T * detection * rng.lognormal(0, 0.15, (rows.size, 1))
    counts = nb(rng, mu)
    names = [s.upper() if i % 27 == 3 else s for i, s in enumerate(symbols())]   # 6 genes upper-case
    n = rows.size
    obs = pd.DataFrame({
        "sample_id": pd.Categorical([sid] * n), "x": x, "y": y, "in_tissue": 1,
        "array_row": rows, "array_col": array_col,
        "total_counts": counts.sum(1), "n_genes_by_counts": (counts > 0).sum(1), "pct_counts_mt": 0.0,
    }, index=[f"{sid}:SPOT{i:04d}-1" for i in range(n)])
    var = pd.DataFrame({"gene_ids": [f"ENSG{i:011d}" for i in range(N_GENES)],
                        "feature_types": "Gene Expression", "mt": False}, index=names)
    C = sp.csr_matrix(counts)
    lib = np.asarray(C.sum(1)).ravel()
    norm = sp.csr_matrix(sp.diags(1e4 / np.maximum(lib, 1)) @ C)
    norm.data = np.log1p(norm.data)
    adata = ad.AnnData(X=norm.astype(np.float32), obs=obs, var=var, layers={"counts": C.astype(np.int32)})
    adata.obsm["spatial"] = np.column_stack([x, y]).astype(np.float64)
    adata.uns["pennsieve"] = {"platform": "visium", "kind": "spot", "coordinate_unit": "pixel",
                              "spot_diameter": 110.0, "grid": "hex", "normalized": True, "target_sum": 1e4}
    if with_image:
        scalef = 0.1
        h, wd = int((y.max() + 1000) * scalef), int((x.max() + 1000) * scalef)
        img = np.full((h, wd, 3), 235, np.uint8)
        img[:, :, 0] = 225
        adata.uns["spatial"] = {sid: {"images": {"lowres": img},
                                      "scalefactors": {"tissue_lowres_scalef": scalef, "tissue_hires_scalef": 0.3,
                                                       "spot_diameter_fullres": 110.0,
                                                       "fiducial_diameter_fullres": 170.0}}}
    adata.obs["planted_follicle"] = follicle
    adata.obsm["planted_abundance"] = w
    return adata


def write(root: Path, layout: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sig = signatures(rng)
    ref = make_reference(rng, sig)
    spatial = [make_spatial(rng, sig, sid, i, i == 0) for i, sid in enumerate(SAMPLES)]
    if root.exists():
        shutil.rmtree(root)
    dirs = {k: root / k for k in ("spatial", "metadata", "reference")} if layout == "linked" else \
        {k: root for k in ("spatial", "metadata", "reference")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    ref.write_h5ad(dirs["reference"] / "reference.h5ad")
    rows = []
    for a in spatial:
        sid = str(a.obs["sample_id"].iloc[0])
        a.write_h5ad(dirs["spatial"] / f"{sid}.h5ad")
        rows.append({"sample_id": sid, "path": sid, "platform": "visium", "kind": "spot", "n_obs": a.n_obs,
                     "n_genes": a.n_vars, "n_obs_in_tissue": a.n_obs, "condition": "lymph"})
    pd.DataFrame(rows).to_csv(dirs["metadata"] / "samples.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--layout", choices=("linked", "flat", "both"), default="both")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    for layout in (("linked", "flat") if a.layout == "both" else (a.layout,)):
        write(a.out / layout if a.layout == "both" else a.out, layout, a.seed)
        print(f"wrote {a.out / layout if a.layout == 'both' else a.out}")


if __name__ == "__main__":
    main()
