"""
cell2location: cell-type deconvolution of spot-based spatial transcriptomics (Visium) with a
single-cell reference, the step after st-qc in the Cell-type Deconvolution & Niches workflow.

Reads the spatial samples (port `spatial`, <sample_id>.h5ad from st-qc or st-prep), samples.csv
(port `metadata`) and a scRNA-seq reference (port `reference`: an .h5ad with raw counts and a
cell type column, or a signatures.csv from an earlier run), and runs the three stages of the
official cell2location tutorial (Kleshchevnikov et al. 2022):

  1. reference signatures: RegressionModel on the reference -> model/signatures.csv
  2. spatial mapping: Cell2location on all samples at once -> labels/abundance.csv (q05)
  3. co-location groups: run_colocation (NMF) on the abundances -> labels/colocation*.csv

and writes

  labels/labels.csv              sample_id, obs_id, dominant_type, colocation_group (strongest NMF group)
  labels/abundance.csv           sample_id, obs_id, one column per cell type (q05 cells per spot)
  labels/proportions.csv         the same, divided by each spot's total
  labels/colocation.csv          group, cell_type, weight
  labels/colocation_by_obs.csv   sample_id, obs_id, one column per group (NMF loadings)
  labels/type_summary.csv        sample_id, cell_type, abundance totals, dominant counts
  labels/summary.json, labels/figures/*.png, labels/provenance.json
  model/signatures.csv, model/reference_summary.csv, model/reference_history.csv,
  model/spatial_history.csv, model/genes_used.csv, model/provenance.json
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("cell2location-app")
log.setLevel(logging.INFO)
for noisy in ("matplotlib", "PIL", "fontTools", "numba", "h5py"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "cell2location"
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "CELL_TYPE_COLUMN": "cell_type",
    "REFERENCE_BATCH_COLUMN": "",
    "MIN_CELLS_PER_TYPE": "20",
    "MAX_CELLS_PER_TYPE": "300",    # cpu variant; the gpu variant sets 2000 (app.yml)
    "CELL_COUNT_CUTOFF": "5",
    "CELL_PERCENTAGE_CUTOFF2": "0.03",
    "NONZ_MEAN_CUTOFF": "1.12",
    "REFERENCE_EPOCHS": "250",
    "N_CELLS_PER_LOCATION": "8",
    "DETECTION_ALPHA": "20",
    "SPATIAL_EPOCHS": "1000",      # cpu variant; the gpu variant sets 30000 (app.yml)
    "MIN_DOMINANT_FRACTION": "0.4",
    "N_COLOCATION_GROUPS": "0",
    "MAX_PLOTS": "8",
    "SEED": "42",
    "THREADS": "0",
    "USE_GPU": "false",           # cpu variant; the gpu variant sets auto
    "ALLOW_LONG_CPU_RUN": "false",  # run on CPU even when the estimate exceeds the CPU budget (processor/budget.py)
}
PARAMETERS = tuple(DEFAULTS)
MIN_GENES_MATCHED = 50          # below this the mapping is not worth running
FEW_GENES_WARNING = 1000


class ParameterError(ValueError):
    pass


def read_parameters(env=os.environ) -> dict:
    from processor.resources import DEVICES, normalise_use_gpu

    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {"CELL_TYPE_COLUMN": raw["CELL_TYPE_COLUMN"].strip(),
               "REFERENCE_BATCH_COLUMN": raw["REFERENCE_BATCH_COLUMN"].strip()}
    if not p["CELL_TYPE_COLUMN"]:
        raise ParameterError("CELL_TYPE_COLUMN must name the reference's cell type column")
    try:
        for k in ("MIN_CELLS_PER_TYPE", "MAX_CELLS_PER_TYPE", "REFERENCE_EPOCHS", "SPATIAL_EPOCHS",
                  "N_COLOCATION_GROUPS", "MAX_PLOTS", "SEED", "THREADS"):
            v = float(raw[k])
            if v != int(v):
                raise ValueError(f"{k} must be a whole number, got {raw[k]!r}")
            p[k] = int(v)
        for k in ("CELL_COUNT_CUTOFF", "CELL_PERCENTAGE_CUTOFF2", "NONZ_MEAN_CUTOFF", "N_CELLS_PER_LOCATION",
                  "DETECTION_ALPHA", "MIN_DOMINANT_FRACTION"):
            p[k] = float(raw[k])
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    if p["MIN_CELLS_PER_TYPE"] < 1:
        raise ParameterError("MIN_CELLS_PER_TYPE must be >= 1")
    if p["MAX_CELLS_PER_TYPE"] < 0:
        raise ParameterError("MAX_CELLS_PER_TYPE must be >= 0 (0 = no limit)")
    if 0 < p["MAX_CELLS_PER_TYPE"] < p["MIN_CELLS_PER_TYPE"]:
        raise ParameterError("MAX_CELLS_PER_TYPE must be 0 or >= MIN_CELLS_PER_TYPE")
    if p["REFERENCE_EPOCHS"] < 1 or p["SPATIAL_EPOCHS"] < 1:
        raise ParameterError("REFERENCE_EPOCHS and SPATIAL_EPOCHS must be >= 1")
    if p["CELL_COUNT_CUTOFF"] < 0 or p["NONZ_MEAN_CUTOFF"] < 0 or not 0 <= p["CELL_PERCENTAGE_CUTOFF2"] <= 1:
        raise ParameterError("CELL_COUNT_CUTOFF and NONZ_MEAN_CUTOFF must be >= 0, CELL_PERCENTAGE_CUTOFF2 in [0, 1]")
    if p["N_CELLS_PER_LOCATION"] <= 0 or p["DETECTION_ALPHA"] <= 0:
        raise ParameterError("N_CELLS_PER_LOCATION and DETECTION_ALPHA must be > 0")
    if not 0 <= p["MIN_DOMINANT_FRACTION"] <= 1:
        raise ParameterError("MIN_DOMINANT_FRACTION is a fraction, 0-1")
    if p["N_COLOCATION_GROUPS"] < 0 or p["N_COLOCATION_GROUPS"] == 1:
        raise ParameterError("N_COLOCATION_GROUPS must be 0 (choose among 4-8) or >= 2")
    if p["MAX_PLOTS"] < 0 or p["THREADS"] < 0:
        raise ParameterError("MAX_PLOTS and THREADS must be >= 0")
    allow = raw["ALLOW_LONG_CPU_RUN"].strip().lower()
    if allow not in ("true", "false", "1", "0", "yes", "no"):
        raise ParameterError(f"ALLOW_LONG_CPU_RUN must be true or false, got {raw['ALLOW_LONG_CPU_RUN']!r}")
    p["ALLOW_LONG_CPU_RUN"] = allow in ("true", "1", "yes")
    p["USE_GPU"] = normalise_use_gpu(raw["USE_GPU"])
    if p["USE_GPU"] not in DEVICES:
        raise ParameterError(f"USE_GPU must be auto, true or false, got {raw['USE_GPU']!r}")
    return p


def _rel(path, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return v


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r.get(c, "")) for c in columns])


def write_frame(path: Path, df, index: bool = False) -> None:
    df.to_csv(path, index=index, float_format="%.6g")


def _clean_json(obj):
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, data) -> None:
    with open(path, "w") as fh:
        json.dump(_clean_json(data), fh, indent=2)
        fh.write("\n")


TOOLS = ("cell2location", "scvi-tools", "torch", "pyro-ppl", "lightning", "scanpy", "anndata", "numpy", "scipy",
         "pandas", "scikit-learn", "h5py", "matplotlib")


def tool_versions() -> dict:
    from importlib.metadata import version

    out = {"python": platform.python_version()}
    for name in TOOLS:
        try:
            out[name] = version(name)
        except Exception:   # noqa: BLE001
            out[name] = None
    return out


def peak_rss_mb() -> float | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except (ImportError, OSError):
        return None


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


CITATIONS = [
    {"name": "cell2location", "doi": "10.1038/s41587-021-01139-4", "pmid": "35027729",
     "text": "Kleshchevnikov V, et al. Cell2location maps fine-grained cell types in spatial transcriptomics. "
             "Nat Biotechnol 2022;40:661-671."},
    {"name": "scvi-tools", "doi": "10.1038/s41587-021-01206-w", "pmid": "35132262",
     "text": "Gayoso A, et al. A Python library for probabilistic analysis of single-cell omics data. "
             "Nat Biotechnol 2022;40:163-166."},
    {"name": "Pyro", "doi": None, "pmid": None,
     "text": "Bingham E, et al. Pyro: Deep Universal Probabilistic Programming. J Mach Learn Res 2019;20(28):1-6."},
    {"name": "scanpy", "doi": "10.1186/s13059-017-1382-0", "pmid": "29409532",
     "text": "Wolf FA, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data analysis. "
             "Genome Biol 2018;19:15."},
    {"name": "anndata", "doi": "10.21105/joss.04371", "pmid": None,
     "text": "Virshup I, et al. anndata: Access and store annotated data matrices. JOSS 2024;9(96):4371."},
    {"name": "NMF (scikit-learn)", "doi": None, "pmid": None,
     "text": "Pedregosa F, et al. Scikit-learn: Machine Learning in Python. J Mach Learn Res 2011;12:2825-2830."},
]


# ---------------------------------------------------------------------------------------------------------------------
# stages


def prepare_reference(found, params: dict, input_dir: Path, fig_dir: Path) -> dict:
    """The reference up to (not including) training: signatures.csv as given, or the .h5ad loaded,
    subsampled and gene-filtered. "genes" is a signatures-like frame (index = symbol, gene_id) of the
    genes the signatures will have, for counting the genes shared with the spatial data."""
    import pandas as pd

    from processor import genes as G
    from processor import reference as R

    t0 = time.monotonic()
    if found.reference_kind == "signatures":
        sig = G.read_signatures(found.reference)
        types = G.signature_types(sig)
        log.info("reference: signatures.csv given (%s): %d genes x %d cell types; skipping the regression model",
                 _rel(found.reference, input_dir), len(sig), len(types))
        return {"kind": "signatures", "signatures": sig, "genes": sig, "types": types, "n_cells": 0, "n_genes": 0,
                "seconds": round(time.monotonic() - t0, 1)}

    ref = R.load_reference(found.reference, params["CELL_TYPE_COLUMN"], params["REFERENCE_BATCH_COLUMN"])
    n_cells_in, n_genes_in, counts_source = ref.n_obs, ref.n_vars, ref.uns["counts_source"]
    ref, summary = R.select_cells(ref, params["MIN_CELLS_PER_TYPE"], params["MAX_CELLS_PER_TYPE"], params["SEED"])
    if ref.obs["cell_type"].nunique() < 2:
        raise ValueError("fewer than 2 cell types left in the reference; lower MIN_CELLS_PER_TYPE")
    ref = R.filter_genes(ref, params["CELL_COUNT_CUTOFF"], params["CELL_PERCENTAGE_CUTOFF2"], params["NONZ_MEAN_CUTOFF"],
                         fig_dir / "reference_gene_filter.png")
    if ref.n_vars < MIN_GENES_MATCHED:
        raise ValueError(f"{ref.n_vars} genes pass filter_genes; relax CELL_COUNT_CUTOFF / CELL_PERCENTAGE_CUTOFF2 / "
                         "NONZ_MEAN_CUTOFF")
    genes = pd.DataFrame({"gene_id": ref.var["gene_id"].astype(str).values}, index=G.make_unique(ref.var["symbol"]))
    if not (genes["gene_id"].str.len() > 0).any():
        genes = genes.drop(columns="gene_id")
    return {"kind": "h5ad", "adata": ref, "summary": summary, "genes": genes, "n_cells": int(ref.n_obs),
            "n_genes": int(ref.n_vars), "n_cells_in": n_cells_in, "n_genes_in": n_genes_in,
            "counts_source": counts_source, "seconds": round(time.monotonic() - t0, 1)}


def stage_reference(prep: dict, params: dict, accelerator: str, found, out_model: Path) -> dict:
    """Signatures from the prepared reference (or as given); writes model/signatures.csv and reference_summary.csv."""
    import pandas as pd

    from processor import genes as G
    from processor import reference as R

    t0 = time.monotonic()
    if prep["kind"] == "signatures":
        sig, types = prep["signatures"], prep["types"]
        shutil.copyfile(found.reference, out_model / "signatures.csv")
        summary = pd.DataFrame({"cell_type": types, "n_cells": "", "n_used": "", "status": "from signatures.csv"})
        write_frame(out_model / "reference_summary.csv", summary)
        return {"signatures": sig, "summary": summary, "history": None, "qc": None, "source": "signatures.csv",
                "seconds": round(prep["seconds"] + time.monotonic() - t0, 1), "n_genes": len(sig),
                "n_types": len(types)}

    ref, summary = prep["adata"], prep["summary"]
    res = R.train_signatures(ref, bool(params["REFERENCE_BATCH_COLUMN"]), params["REFERENCE_EPOCHS"], accelerator,
                             params["SEED"])
    sig = res["signatures"]
    write_frame(out_model / "signatures.csv", sig, index=True)
    write_frame(out_model / "reference_summary.csv", summary)
    write_frame(out_model / "reference_history.csv", res["history"])
    return {"signatures": sig, "summary": summary, "history": res["history"], "qc": res["qc"], "source": "h5ad",
            "seconds": round(prep["seconds"] + time.monotonic() - t0, 1), "seconds_train": res["seconds_train"],
            "final_elbo": res["final_elbo"], "n_batches": res["n_batches"], "counts_source": prep["counts_source"],
            "n_cells_in": prep["n_cells_in"], "n_genes_in": prep["n_genes_in"], "n_cells_used": int(ref.n_obs),
            "n_genes": int(ref.n_vars), "n_types": int(len(G.signature_types(sig))),
            "log1p_correlation": res["qc"]["log1p_correlation"]}


def load_samples(found, input_dir: Path) -> tuple[list, dict, list[dict]]:
    """Slim copies of every readable sample; the images for plotting; a status per sample."""
    from processor import mapping as M

    adatas, images, statuses = [], {}, []
    for sid, path in found.spatial:
        t = time.monotonic()
        status = {"sample_id": sid, "file": _rel(path, input_dir)}
        try:
            a, image = M.load_sample(path, sid)
            if a.n_obs == 0:
                raise ValueError("no observations")
            sids = set(a.obs["sample_id"].astype(str))
            if sids != {sid}:
                log.warning("%s: obs['sample_id'] is %s; using the file name", sid, ", ".join(sorted(sids)))
                a.obs["sample_id"] = sid
            adatas.append(a)
            images[sid] = image
            n_counts = float(a.X.sum())
            status.update(status="ok", n_obs=int(a.n_obs), n_genes=int(a.n_vars), counts_source=a.uns["counts_source"],
                          median_counts=float(_median_counts(a)), has_image=image is not None)
            log.info("%s: %d observations x %d genes, counts from %s (%.3g in total), image %s, %.1fs", sid, a.n_obs,
                     a.n_vars, a.uns["counts_source"], n_counts, image["key"] if image else "none",
                     time.monotonic() - t)
        except Exception as exc:   # one bad sample is skipped, not fatal
            status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            log.error("%s: skipped: %s", sid, exc)
        statuses.append(status)
    return adatas, images, statuses


def _median_counts(a) -> float:
    import numpy as np

    return float(np.median(np.asarray(a.X.sum(1)).ravel()))


def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 2
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        params = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2

    from processor import resources as RS

    threads, threads_source = RS.resolve_threads(params["THREADS"])
    RS.set_thread_env(threads)    # no-op for numpy when entry.py has already done it; torch reads it below
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import torch

    from processor import budget as B
    from processor import colocation as CO
    from processor import figures as F
    from processor import mapping as M
    from processor import training as T
    from processor.inputs import discover

    T.quiet_libraries()
    torch.set_num_threads(threads)
    device = RS.choose_device(params["USE_GPU"])
    log.info("parameters: %s", params)
    log.info("threads: %d (%s); %s", threads, threads_source, device["reason"])
    if params["USE_GPU"] == "true" and device["accelerator"] == "cpu":
        log.warning("USE_GPU=true but no usable GPU; running on cpu (much slower)")
    accelerator = device["accelerator"]

    try:
        found = discover(input_dir)
    except (OSError, ValueError) as exc:
        log.error("cannot read the inputs: %s", exc)
        return 1
    for name, reason in found.skipped.items():
        log.warning("%s: skipped, %s", name, reason)
    if found.reference is None:
        log.error("no reference found: link a scRNA-seq .h5ad (or a signatures.csv) to the reference input")
        return 1
    if not found.spatial:
        log.error("no spatial <sample_id>.h5ad found under %s", input_dir)
        return 1
    out_labels, out_model = output_dir / "labels", output_dir / "model"
    fig_dir = out_labels / "figures"
    for d in (out_labels, out_model, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    log.info("%d spatial samples, reference %s (%s), samples.csv %s", len(found.spatial),
             _rel(found.reference, input_dir), found.reference_kind, _rel(found.samples_csv, input_dir))
    timings: dict[str, float] = {}
    figures: list[str] = []
    rss: dict[str, float | None] = {}

    def provenance(status: str, extra: dict) -> dict:
        return {
            "app": APP_NAME, "version": app_version(), "status": status, "tools": tool_versions(),
            "parameters": params, "threads": {"n": threads, "source": threads_source}, "device": device,
            "inputs": {"spatial": [_rel(p, input_dir) for _, p in found.spatial],
                       "reference": _rel(found.reference, input_dir), "reference_kind": found.reference_kind,
                       "samples_csv": _rel(found.samples_csv, input_dir), "skipped": found.skipped},
            "method": method_text(params),
            "citations": CITATIONS,
            "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
            "timings_seconds": {**timings, "total": round(time.monotonic() - t0, 1)},
            "peak_rss_mb": peak_rss_mb(), "peak_rss_mb_by_stage": rss,
            "cgroup_memory_peak_mb": RS.cgroup_memory_peak_mb(), **extra,
        }

    # ---- stage 1a: the reference up to training
    try:
        prep = prepare_reference(found, params, input_dir, fig_dir)
    except Exception as exc:
        log.exception("reference signatures failed: %s", exc)
        write_json(out_model / "provenance.json", provenance("failed", {"error": f"reference: {exc}"}))
        return 1

    # ---- the spatial samples, and the genes they share with the reference
    t = time.monotonic()
    adatas, images, statuses = load_samples(found, input_dir)
    if not adatas:
        log.error("every spatial sample failed to load")
        write_json(out_model / "provenance.json", provenance("failed", {"samples": statuses}))
        return 1
    try:
        vis = M.concat_samples(adatas)
        del adatas
        n_shared = M.count_shared_genes(vis, prep["genes"])
    except Exception as exc:
        log.exception("spatial mapping failed: %s", exc)
        write_json(out_model / "provenance.json", provenance("failed", {"error": f"mapping: {exc}", "samples": statuses}))
        return 1
    timings["load_samples"] = round(time.monotonic() - t, 1)

    # ---- the CPU size guard, before any training (processor/budget.py)
    cpu_estimate = B.estimate_cpu_seconds(vis.n_obs, n_shared, params["SPATIAL_EPOCHS"], prep["n_cells"],
                                          prep["n_genes"], params["REFERENCE_EPOCHS"] if prep["kind"] == "h5ad" else 0)
    verdict, message = B.check(cpu_estimate, accelerator, params["ALLOW_LONG_CPU_RUN"])
    cpu_estimate["verdict"] = verdict
    if verdict == "refused":
        log.error("%s", message)
        write_json(out_model / "provenance.json", provenance("refused", {"error": message, "cpu_estimate": cpu_estimate,
                                                                         "samples": statuses}))
        return B.EXIT_TOO_LARGE_FOR_CPU
    (log.warning if verdict == "allowed" else log.info)("%s", message)

    # ---- stage 1b: reference signatures
    try:
        ref = stage_reference(prep, params, accelerator, found, out_model)
    except Exception as exc:
        log.exception("reference signatures failed: %s", exc)
        write_json(out_model / "provenance.json", provenance("failed", {"error": f"reference: {exc}"}))
        return 1
    del prep
    timings["reference"] = ref["seconds"]
    rss["reference"] = peak_rss_mb()

    # ---- stage 2: spatial mapping
    t = time.monotonic()
    try:
        vis, cell_state, genes_used, gene_counts = M.intersect(vis, ref["signatures"])
        write_frame(out_model / "genes_used.csv", genes_used)
        if len(genes_used) < MIN_GENES_MATCHED:
            raise ValueError(f"only {len(genes_used)} genes shared by the signatures and the spatial data "
                             f"(need {MIN_GENES_MATCHED}); is the reference from the same species?")
        if len(genes_used) < FEW_GENES_WARNING:
            log.warning("only %d genes shared by the signatures and the spatial data; abundances will be imprecise",
                        len(genes_used))
        timings["intersect"] = round(time.monotonic() - t, 1)
        mapped = M.train_mapping(vis, cell_state, params["N_CELLS_PER_LOCATION"], params["DETECTION_ALPHA"],
                                 params["SPATIAL_EPOCHS"], accelerator, params["SEED"])
    except Exception as exc:
        log.exception("spatial mapping failed: %s", exc)
        write_json(out_model / "provenance.json", provenance("failed", {"error": f"mapping: {exc}", "samples": statuses}))
        return 1
    timings["mapping"] = mapped["seconds"]
    rss["mapping"] = peak_rss_mb()
    vis = mapped["adata"]
    write_frame(out_model / "spatial_history.csv", mapped["history"])

    q05, _means = M.abundance_tables(vis)
    types = list(q05.columns)
    sample_of = vis.obs["sample_id"].astype(str)
    dominant, dom_frac = M.dominant_types(q05, params["MIN_DOMINANT_FRACTION"])
    ids = pd.DataFrame({"sample_id": sample_of.values, "obs_id": q05.index.astype(str)}, index=q05.index)
    write_frame(out_labels / "abundance.csv", pd.concat([ids, q05], axis=1))
    totals = q05.sum(axis=1)
    props = q05.div(totals.replace(0, np.nan), axis=0).fillna(0.0)
    write_frame(out_labels / "proportions.csv", pd.concat([ids, props], axis=1))

    # ---- stage 3: co-location groups
    t = time.monotonic()
    coloc = None
    try:
        coloc = CO.run(vis, params["N_COLOCATION_GROUPS"], params["SEED"])
        write_frame(out_labels / "colocation.csv", coloc["colocation"])
        by_obs = coloc["by_obs"]
        write_frame(out_labels / "colocation_by_obs.csv", pd.concat([ids, by_obs.loc[ids.index]], axis=1))
    except Exception as exc:   # the abundances stand without the co-location groups
        log.exception("co-location NMF failed: %s", exc)
    # labels.csv: the dominant cell type, and (when the NMF ran) each spot's strongest co-location
    # group. With a fine-grained reference a spot rarely has one dominant type (most are 'Mixed'),
    # while every spot has a strongest group, so colocation_group is the label for niche analysis.
    labels = ids.assign(dominant_type=dominant.values)
    if coloc is not None:
        g = coloc["by_obs"].loc[ids.index]
        labels["colocation_group"] = g.idxmax(axis=1).where(g.max(axis=1) > 0, "None").values
    write_frame(out_labels / "labels.csv", labels)
    timings["colocation"] = round(time.monotonic() - t, 1)
    rss["colocation"] = peak_rss_mb()

    # ---- per-sample summaries
    type_rows, sample_rows = [], []
    for s in [x for x in statuses if x["status"] == "ok"]:
        sid = s["sample_id"]
        m = (sample_of == sid).to_numpy()
        q = q05[m]
        tot = float(q.to_numpy().sum())
        dom_s = dominant[m]
        n = int(m.sum())
        for ct in types:
            n_dom = int((dom_s == ct).sum())
            type_rows.append({"sample_id": sid, "cell_type": ct, "total_abundance": float(q[ct].sum()),
                              "mean_abundance": float(q[ct].mean()), "fraction_of_cells": float(q[ct].sum() / tot) if tot else 0.0,
                              "n_dominant": n_dom, "fraction_dominant": n_dom / n if n else 0.0})
        top = q.sum().sort_values(ascending=False)
        dom_counts = dom_s.value_counts()
        s.update(n_mapped=n, mean_cells_per_obs=round(float(totals[m].mean()), 3),
                 n_mixed=int(dom_counts.get("Mixed", 0)),
                 top_types=[f"{ct} ({v / tot:.3f})" for ct, v in top.head(5).items()] if tot else [],
                 dominant_counts={str(k): int(v) for k, v in dom_counts.items()})
        sample_rows.append(s)
        log.info("%s: %d spots, %.1f cells per spot (q05), %d Mixed; top types %s; dominant %s", sid, n,
                 s["mean_cells_per_obs"], s["n_mixed"], ", ".join(s["top_types"][:3]),
                 ", ".join(f"{k} {v}" for k, v in list(s["dominant_counts"].items())[:4]))
    type_summary = pd.DataFrame(type_rows, columns=["sample_id", "cell_type", "total_abundance", "mean_abundance",
                                                    "fraction_of_cells", "n_dominant", "fraction_dominant"])
    write_frame(out_labels / "type_summary.csv", type_summary)

    # ---- figures
    t = time.monotonic()
    colors = F.type_colors(types)
    try:
        figures.append(F.training_history({"reference (RegressionModel)": ref["history"],
                                           "spatial (Cell2location)": mapped["history"]},
                                          fig_dir / "training_history.png").name)
        figures.append(F.reconstruction(ref["qc"], mapped["qc"], fig_dir / "reconstruction_qc.png").name)
        figures.append(F.type_composition(type_summary, colors, fig_dir / "type_composition.png").name)
        if coloc is not None:
            figures.append(F.colocation_heatmap(coloc["colocation"], fig_dir / "colocation_weights.png").name)
    except Exception as exc:   # figures are a convenience
        log.exception("figures failed: %s", exc)
    if (fig_dir / "reference_gene_filter.png").exists():
        figures.append("reference_gene_filter.png")
    xy_all = np.asarray(vis.obsm["spatial"])
    for s in sample_rows:
        sid = s["sample_id"]
        m = (sample_of == sid).to_numpy()
        try:
            xy, img = xy_all[m], images.get(sid)
            if params["MAX_PLOTS"] > 0:
                top = q05[m].sum().sort_values(ascending=False).index[: params["MAX_PLOTS"]].tolist()
                figures.append(F.abundance_maps(sid, xy, q05[m], top, img, fig_dir / f"{sid}_abundance.png").name)
            figures.append(F.dominant_map(sid, xy, dominant[m], colors, img, params["MIN_DOMINANT_FRACTION"],
                                          fig_dir / f"{sid}_dominant_type.png").name)
            if coloc is not None:
                tops = {g["group"]: [x.split(" (")[0] for x in g["top_types"]] for g in coloc["group_summary"]}
                figures.append(F.colocation_maps(sid, xy, coloc["by_obs"].loc[q05.index[m]], tops, img,
                                                 fig_dir / f"{sid}_colocation.png").name)
        except Exception as exc:
            log.exception("%s: figures failed: %s", sid, exc)
    timings["figures"] = round(time.monotonic() - t, 1)

    # ---- summary, provenance
    n_ok = len(sample_rows)
    n_failed = len(statuses) - n_ok
    reference_info = {k: ref.get(k) for k in ("source", "counts_source", "n_cells_in", "n_genes_in", "n_cells_used",
                                              "n_genes", "n_types", "n_batches", "final_elbo", "log1p_correlation")}
    reference_info["dropped_types"] = ref["summary"].loc[ref["summary"]["status"] != "used", "cell_type"].tolist() \
        if ref["source"] == "h5ad" else []
    summary = {
        "n_samples": len(statuses), "n_ok": n_ok, "n_failed": n_failed, "n_obs": int(vis.n_obs),
        "cell_types": types, "n_cell_types": len(types), "reference": reference_info,
        "genes": gene_counts,
        "mapping": {"n_genes": int(vis.n_vars), "epochs": params["SPATIAL_EPOCHS"], "final_elbo": mapped["final_elbo"],
                    "log1p_correlation": mapped["qc"]["log1p_correlation"],
                    "mean_cells_per_obs": round(float(totals.mean()), 3), "device": accelerator},
        "dominant_type_counts": {str(k): int(v) for k, v in dominant.value_counts().items()},
        "n_mixed": int((dominant == "Mixed").sum()), "min_dominant_fraction": params["MIN_DOMINANT_FRACTION"],
        "colocation_group_counts": {} if coloc is None else {str(k): int(v) for k, v in labels["colocation_group"].value_counts().items()},
        "colocation": None if coloc is None else {
            "n_groups": coloc["n_groups"], "reason": coloc["reason"], "candidates": coloc["candidates"],
            "groups": coloc["group_summary"]},
        "samples": sample_rows, "figures": sorted(figures), "parameters": params,
    }
    write_json(out_labels / "summary.json", summary)
    extra = {"samples": statuses, "n_ok": n_ok, "n_failed": n_failed, "cpu_estimate": cpu_estimate, "summary": {
        "n_obs": summary["n_obs"], "n_cell_types": len(types), "genes": gene_counts,
        "colocation_groups": None if coloc is None else coloc["n_groups"]},
        "outputs": {"labels": sorted(p.name for p in out_labels.iterdir()) + ["provenance.json"],
                    "model": sorted(p.name for p in out_model.iterdir()) + ["provenance.json"]}}
    prov = provenance("ok", extra)
    write_json(out_labels / "provenance.json", prov)
    write_json(out_model / "provenance.json", prov)
    log.info("done: %d samples (%d failed), %d spots, %d cell types, %d genes, %s co-location groups, %.0fs, "
             "peak RSS %s MB, cgroup peak %s MB -> %s", n_ok, n_failed, vis.n_obs, len(types), vis.n_vars,
             "no" if coloc is None else coloc["n_groups"], time.monotonic() - t0, prov["peak_rss_mb"],
             prov["cgroup_memory_peak_mb"], output_dir)
    return 0


def method_text(p: dict) -> dict:
    return {
        "reference": (f"cells of types with < MIN_CELLS_PER_TYPE={p['MIN_CELLS_PER_TYPE']} cells dropped, the rest "
                      f"subsampled (seeded) to MAX_CELLS_PER_TYPE={p['MAX_CELLS_PER_TYPE']}; "
                      f"cell2location.utils.filtering.filter_genes(cell_count_cutoff={p['CELL_COUNT_CUTOFF']:g}, "
                      f"cell_percentage_cutoff2={p['CELL_PERCENTAGE_CUTOFF2']:g}, nonz_mean_cutoff={p['NONZ_MEAN_CUTOFF']:g}); "
                      "RegressionModel.setup_anndata(batch_key=REFERENCE_BATCH_COLUMN or one batch, labels_key=cell type); "
                      f"train(max_epochs={p['REFERENCE_EPOCHS']}, batch_size=2500, train_size=1, lr=0.002); "
                      "signatures = posterior means_per_cluster_mu_fg (1000 samples). Skipped when a signatures.csv is given."),
        "mapping": ("raw counts (layers['counts'] or X) of every sample concatenated; mitochondrial genes left out; genes "
                    "matched to the signatures by Ensembl id, then symbol, then symbol ignoring case; "
                    "Cell2location.setup_anndata(batch_key='sample_id'); "
                    f"Cell2location(N_cells_per_location={p['N_CELLS_PER_LOCATION']:g}, detection_alpha={p['DETECTION_ALPHA']:g}); "
                    f"train(max_epochs={p['SPATIAL_EPOCHS']}, batch_size=None, train_size=1); "
                    "abundance = q05 of the posterior cell abundance (1000 samples)"),
        "dominant_type": (f"cell type with the largest q05 abundance; 'Mixed' when it holds < MIN_DOMINANT_FRACTION="
                          f"{p['MIN_DOMINANT_FRACTION']:g} of the spot's total q05 abundance"),
        "colocation": ("cell2location CoLocatedGroupsSklearnNMF (the run_colocation model: init='random', l1_ratio=0.5, "
                       "n_iter=20000, 5 restarts) on the q05 abundances, regularisation alpha=0.01 passed as "
                       "alpha_W=0.01/n_cell_types, alpha_H=0.01/n_locations (scikit-learn >= 1.2 scales alpha); "
                       + (f"N_COLOCATION_GROUPS={p['N_COLOCATION_GROUPS']}" if p["N_COLOCATION_GROUPS"] else
                          "4-8 groups tried, the largest reproducible on 5 seeded 80 % subsamples of the locations "
                          "(worst-matched group's cell-type loadings correlate >= 0.9 on average) kept")),
    }


if __name__ == "__main__":
    sys.exit(main())
