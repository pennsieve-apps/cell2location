"""
Figures (PNG, 150 dpi). Spatial plots draw obsm["spatial"] with y inverted (image convention),
on the lowres H&E when the sample has one (coordinates scaled by tissue_lowres_scalef), with
spots drawn at their real diameter when it is known.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 150
MIXED = "Mixed"


def type_colors(types: list[str]) -> dict[str, tuple]:
    base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("Set3").colors)
    out = {t: base[i % len(base)] for i, t in enumerate(types)}
    out[MIXED] = (0.75, 0.75, 0.75)
    return out


def _coords(xy: np.ndarray, image: dict | None) -> tuple[np.ndarray, float | None]:
    """Plot coordinates (scaled onto the image when there is one) and the spot diameter in them."""
    if image is not None:
        s = image["scalef"]
        d = image.get("spot_diameter_fullres") or 0
        return xy * s, (d * s if d else None)
    return xy, None


def _spot_size(ax, xy: np.ndarray, diameter: float | None) -> float:
    """Marker size (points^2) for a spot `diameter` data units wide; falls back to the median neighbour spacing."""
    if diameter is None or diameter <= 0:
        if len(xy) > 1:
            from scipy.spatial import cKDTree

            d, _ = cKDTree(xy).query(xy, k=2)
            diameter = float(np.median(d[:, 1])) * 0.95
        else:
            diameter = 1.0
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    x0, x1 = ax.get_xlim()
    pts_per_unit = bbox.width * 72 / abs(x1 - x0)
    return max(0.5, (diameter * pts_per_unit) ** 2)


def _panel(ax, xy, image, values=None, colors=None, cmap="magma", vmax=None, title=""):
    pxy, diameter = _coords(xy, image)
    pad = 0.04 * max(np.ptp(pxy[:, 0]), np.ptp(pxy[:, 1]), 1.0)
    ax.set_xlim(pxy[:, 0].min() - pad, pxy[:, 0].max() + pad)
    ax.set_ylim(pxy[:, 1].max() + pad, pxy[:, 1].min() - pad)       # y inverted
    ax.set_aspect("equal")
    if image is not None:
        ax.imshow(image["image"], zorder=0)
        ax.set_xlim(pxy[:, 0].min() - pad, pxy[:, 0].max() + pad)
        ax.set_ylim(pxy[:, 1].max() + pad, pxy[:, 1].min() - pad)
    size = _spot_size(ax, pxy, diameter)
    if colors is not None:
        sc = ax.scatter(pxy[:, 0], pxy[:, 1], c=colors, s=size, linewidths=0, zorder=1)
    else:
        v = np.asarray(values, dtype=float)
        hi = vmax if vmax is not None else (np.nanquantile(v, 0.99) if len(v) else 1.0)
        sc = ax.scatter(pxy[:, 0], pxy[:, 1], c=v, s=size, cmap=cmap, vmin=0, vmax=max(hi, 1e-9), linewidths=0,
                        zorder=1)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def _grid(n: int, ncol: int = 4, size: float = 3.4):
    ncol = min(ncol, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes.ravel()


def abundance_maps(sid: str, xy: np.ndarray, q05: pd.DataFrame, types: list[str], image: dict | None,
                   path: Path) -> Path:
    """q05 cell abundance of the given types, one panel each."""
    fig, axes = _grid(len(types))
    for ax, t in zip(axes, types):
        _panel(ax, xy, image, values=q05[t].to_numpy(), title=f"{t}\n(q05 cells per spot)")
    fig.suptitle(f"{sid}: estimated cell abundance (cell2location q05), top {len(types)} cell types", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def dominant_map(sid: str, xy: np.ndarray, labels: pd.Series, colors: dict, image: dict | None,
                 min_fraction: float, path: Path) -> Path:
    fig, (ax, lax) = plt.subplots(1, 2, figsize=(10.5, 6), gridspec_kw={"width_ratios": [3, 1.3]})
    _panel(ax, xy, image, colors=[colors.get(v, (0, 0, 0)) for v in labels])
    ax.set_title(f"{sid}: dominant cell type per spot\n({MIXED}: top type < {min_fraction:g} of the spot's cells)",
                 fontsize=10)
    counts = labels.value_counts()
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors.get(t, (0, 0, 0)), label=f"{t} ({n})")
               for t, n in counts.items()]
    lax.legend(handles=handles, loc="center left", frameon=False, fontsize=7)
    lax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def colocation_maps(sid: str, xy: np.ndarray, by_obs: pd.DataFrame, top_types: dict[str, list[str]],
                    image: dict | None, path: Path) -> Path:
    groups = list(by_obs.columns)
    fig, axes = _grid(len(groups))
    for ax, g in zip(axes, groups):
        _panel(ax, xy, image, values=by_obs[g].to_numpy(), cmap="viridis",
               title=f"{g}: " + ", ".join(top_types.get(g, [])[:3]))
        ax.title.set_fontsize(7)
    fig.suptitle(f"{sid}: co-location groups (NMF location loadings)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def colocation_heatmap(colocation: pd.DataFrame, path: Path) -> Path:
    m = colocation.pivot(index="cell_type", columns="group", values="weight")
    order = m.idxmax(axis=1).map({g: i for i, g in enumerate(m.columns)}).sort_values(kind="stable").index
    m = m.loc[order]
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * m.shape[1] + 3.2, 1.2 + 0.28 * m.shape[0]))
    im = ax.imshow(m.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(m.shape[1]), m.columns)
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=7)
    ax.set_title("Co-location groups: fraction of each cell type's abundance per group", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def training_history(histories: dict[str, pd.DataFrame], path: Path) -> Path:
    items = [(k, h) for k, h in histories.items() if h is not None and len(h)]
    fig, axes = plt.subplots(1, max(1, len(items)), figsize=(5.2 * max(1, len(items)), 3.8), squeeze=False)
    for ax, (label, h) in zip(axes.ravel(), items):
        skip = int(len(h) * 0.1) if len(h) > 20 else 0      # the tutorial drops the first 10 % (the steep start)
        ax.plot(h["epoch"].to_numpy()[skip:], h["elbo"].to_numpy()[skip:], lw=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("-ELBO loss")
        ax.set_title(f"{label} ({len(h)} epochs, first {skip} not shown)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _hexbin(ax, x, y, xlabel, ylabel, title):
    x, y = np.log1p(np.asarray(x, float)), np.log1p(np.asarray(y, float))
    ax.hexbin(x, y, gridsize=60, bins="log", cmap="Greys", mincnt=1)
    hi = float(max(x.max(initial=1), y.max(initial=1)))
    ax.plot([0, hi], [0, hi], color="tab:red", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)


def reconstruction(qc_reference: dict | None, qc_spatial: dict, path: Path) -> Path:
    """The plot_QC panels: observed vs expected counts (both models) and signatures vs plain cluster means."""
    panels = []
    if qc_reference is not None:
        panels.append(("reference", qc_reference))
    panels.append(("spatial", qc_spatial))
    n = len(panels) + (1 if qc_reference is not None and "cluster_means" in qc_reference else 0)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), squeeze=False)
    axes = axes.ravel()
    for ax, (label, qc) in zip(axes, panels):
        _hexbin(ax, qc["observed"], qc["expected"], "observed counts, log1p", "expected counts, log1p",
                f"{label} model: r = {qc['log1p_correlation']:.3f} ({qc['n_obs']} obs)")
    if n > len(panels):
        _hexbin(axes[-1], qc_reference["cluster_means"], qc_reference["estimated"], "mean counts per cell type, log1p",
                "estimated signature, log1p", "signatures vs plain cluster means")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def type_composition(type_summary: pd.DataFrame, colors: dict, path: Path) -> Path:
    """Stacked bars: each sample's share of the estimated cells per type."""
    m = type_summary.pivot(index="sample_id", columns="cell_type", values="fraction_of_cells").fillna(0)
    order = m.mean().sort_values(ascending=False).index
    m = m[order]
    fig, ax = plt.subplots(figsize=(3 + 0.9 * len(m), 5.2))
    bottom = np.zeros(len(m))
    for t in m.columns:
        ax.bar(range(len(m)), m[t].to_numpy(), bottom=bottom, color=colors.get(t), label=t, width=0.75)
        bottom += m[t].to_numpy()
    ax.set_xticks(range(len(m)), m.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("fraction of estimated cells (q05)")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, frameon=False)
    ax.set_title("Cell type composition per sample", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
