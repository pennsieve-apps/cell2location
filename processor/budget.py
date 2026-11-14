"""
How long a run would take on CPU, and the guard that refuses CPU runs too large to be worth it.

cell2location is meant for a GPU: on CPU both models scale with (observations x genes x epochs), and a
full Visium section takes many hours. Before any training, main.py estimates the CPU training time
from the sizes it has just read and, when the run is on CPU and the estimate exceeds
CPU_BUDGET_SECONDS, stops with EXIT_TOO_LARGE_FOR_CPU and a message pointing to the gpu variant,
a smaller input or ALLOW_LONG_CPU_RUN=true. A GPU run is never checked.

Calibration (conservative, from prod on 2026-11-13: the Pennsieve Test compute node, Fargate
4 vCPU / 16 GiB, amd64, the cpu variant v0.1.2 on the lymph node demo):

- spatial model: ~6 s/epoch at 3,928 spots x 17,375 shared genes (20 cell types)
  -> 6 / (3,928 x 17,375) = 8.8e-8 s per spot x gene x epoch; SPATIAL_RATE rounds up to 1.0e-7
  (~6.8 s/epoch for the demo).
- reference model: the reference stage including its preparation took ~80 min for 7,269 cells x
  24,117 genes x 250 epochs -> 4,800 / (7,269 x 24,117 x 250) = 1.1e-7 s per cell x gene x epoch.
  The whole stage is attributed to training, so REFERENCE_RATE over-estimates the training alone
  (locally, native arm64, the same training took 429 s, 9.8e-9).

Both rates are for 4 CPU threads; more threads make a run somewhat faster, not proportionally.
Fixed costs (posterior sampling, NMF, figures) are left out; CPU_BUDGET_SECONDS (2 h) is a quarter
of the cpu variant's 8 h timeout, which leaves room for them and for slower hosts.
"""
from __future__ import annotations

SPATIAL_RATE = 1.0e-7          # seconds per spot x shared gene x epoch (Cell2location, 4 vCPU Fargate)
REFERENCE_RATE = 1.1e-7        # seconds per reference cell x gene x epoch (RegressionModel, 4 vCPU Fargate)
CPU_BUDGET_SECONDS = 2 * 3600  # a quarter of the cpu variant's timeoutSeconds (28800)
EXIT_TOO_LARGE_FOR_CPU = 3


def estimate_cpu_seconds(n_spots: int, n_genes: int, spatial_epochs: int, n_ref_cells: int = 0,
                         n_ref_genes: int = 0, reference_epochs: int = 0) -> dict:
    """Estimated CPU training seconds: {reference, spatial, total} plus the sizes they come from.

    n_genes: genes shared by the signatures and the spatial data; n_ref_cells / n_ref_genes: the
    reference after subsampling and filter_genes (0 when a signatures.csv is given)."""
    reference = float(n_ref_cells) * n_ref_genes * reference_epochs * REFERENCE_RATE
    spatial = float(n_spots) * n_genes * spatial_epochs * SPATIAL_RATE
    return {"n_spots": int(n_spots), "n_genes": int(n_genes), "spatial_epochs": int(spatial_epochs),
            "n_ref_cells": int(n_ref_cells), "n_ref_genes": int(n_ref_genes),
            "reference_epochs": int(reference_epochs),
            "reference_seconds": round(reference), "spatial_seconds": round(spatial),
            "total_seconds": round(reference + spatial),
            "spatial_rate": SPATIAL_RATE, "reference_rate": REFERENCE_RATE, "budget_seconds": CPU_BUDGET_SECONDS}


def _hours(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"


def check(estimate: dict, accelerator: str, allow_long: bool, budget: float | None = None) -> tuple[str, str]:
    """(verdict, message). verdict: gpu (not checked), ok (within the budget), allowed (over it, but
    allow_long) or refused (over it on CPU: the run must stop)."""
    budget = CPU_BUDGET_SECONDS if budget is None else budget
    total = estimate["total_seconds"]
    what = (f"estimated CPU training time {_hours(total)} (reference {_hours(estimate['reference_seconds'])}: "
            f"{estimate['n_ref_cells']} cells x {estimate['n_ref_genes']} genes x {estimate['reference_epochs']} epochs; "
            f"spatial {_hours(estimate['spatial_seconds'])}: {estimate['n_spots']} spots x {estimate['n_genes']} genes x "
            f"{estimate['spatial_epochs']} epochs)")
    if accelerator != "cpu":
        return "gpu", f"running on GPU; {what} does not apply"
    if total <= budget:
        return "ok", f"{what}, within the CPU budget of {_hours(budget)}"
    if allow_long:
        return "allowed", (f"{what}, over the CPU budget of {_hours(budget)}; running anyway because "
                      "ALLOW_LONG_CPU_RUN=true (the step may hit its timeout)")
    return "refused", (f"{what} exceeds the CPU budget of {_hours(budget)}: the CPU variant is for very small data and "
                   "testing. Use the GPU variant of cell2location (the normal choice), or make the input smaller "
                   "(fewer spots or samples, a signatures.csv instead of a reference .h5ad, lower SPATIAL_EPOCHS, "
                   "REFERENCE_EPOCHS or MAX_CELLS_PER_TYPE), or set ALLOW_LONG_CPU_RUN=true to run it on CPU anyway")
