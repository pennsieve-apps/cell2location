"""
Shared pieces of the two scvi-tools / pyro training stages: seeding, a progress log line every
so many epochs (the platform shows stdout, not a progress bar), and the reconstruction QC
numbers (observed counts vs the model's expected counts on a random subset of observations).
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
from lightning.pytorch.callbacks import Callback
from scipy.sparse import issparse

log = logging.getLogger("cell2location-app")


def quiet_libraries() -> None:
    import scvi

    scvi.settings.verbosity = logging.WARNING
    for name in ("cell2location", "scvi", "lightning", "lightning.pytorch", "pytorch_lightning", "lightning_fabric", "pyro", "jax",
                 "matplotlib", "PIL", "fontTools", "numba", "h5py"):
        logging.getLogger(name).setLevel(logging.WARNING)


def seed_everything(seed: int) -> None:
    import scvi
    import torch

    scvi.settings.seed = seed       # numpy, torch and pyro
    torch.manual_seed(seed)
    np.random.seed(seed)


class EpochLog(Callback):
    """One log line every `every` epochs (and at the end) with the training ELBO."""

    def __init__(self, label: str, total: int, every: int):
        self.label, self.total, self.every = label, total, max(1, every)
        self.t0 = time.monotonic()

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every and epoch != self.total:
            return
        elbo = trainer.callback_metrics.get("elbo_train")
        if elbo is None:
            elbo = trainer.logged_metrics.get("elbo_train")
        value = f"{float(elbo):.5g}" if elbo is not None else "n/a"
        log.info("%s: epoch %d/%d, ELBO %s, %.0fs", self.label, epoch, self.total, value, time.monotonic() - self.t0)


def log_every(total: int) -> int:
    """About 10-20 progress lines per training, at round epoch numbers (1, 2, 5 x 10^k)."""
    raw = max(1.0, total / 15)
    k = 10 ** math.floor(math.log10(raw))
    return int(min((m * k for m in (1, 2, 5, 10) if m * k >= raw), default=k))


def train_kwargs(label: str, epochs: int, accelerator: str) -> dict:
    return {"max_epochs": int(epochs), "accelerator": accelerator, "device": "auto",
            "enable_progress_bar": False, "enable_model_summary": False, "enable_checkpointing": False, "callbacks": [EpochLog(label, int(epochs), log_every(epochs))]}


def history_table(mod):
    """mod.history["elbo_train"] as a DataFrame with columns epoch, elbo."""
    import pandas as pd

    h = mod.history.get("elbo_train") if isinstance(mod.history, dict) else None
    if h is None:
        return pd.DataFrame(columns=["epoch", "elbo"])
    out = h.reset_index()
    out.columns = ["epoch", "elbo"]
    return out


def reconstruction_qc(mod, summary: str = "means", n_obs: int = 1000, seed: int = 0) -> dict:
    """Observed counts and the model's expected counts (flattened) for n_obs random observations."""
    from scvi import REGISTRY_KEYS

    rng = np.random.default_rng(seed)
    n = mod.adata_manager.adata.n_obs
    ind = np.sort(rng.choice(n, min(n_obs, n), replace=False))
    expected = mod.module.model.compute_expected(mod.samples[f"post_sample_{summary}"], mod.adata_manager, ind_x=ind)
    x = mod.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)[ind, :]
    if issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64).ravel()
    mu = np.asarray(expected["mu"], dtype=np.float64).ravel()
    ok = np.isfinite(mu) & np.isfinite(x)
    x, mu = x[ok], mu[ok]
    corr = float(np.corrcoef(np.log1p(x), np.log1p(mu))[0, 1]) if len(x) > 1 else float("nan")
    return {"observed": x, "expected": mu, "n_obs": int(len(ind)), "log1p_correlation": round(corr, 4)}
