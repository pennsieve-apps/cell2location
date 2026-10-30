"""
Finding the inputs.

Linked layout (placement: subfolder): INPUT_DIR/spatial/<sample_id>.h5ad (st-qc or st-prep),
INPUT_DIR/metadata/samples.csv and INPUT_DIR/reference/{<name>.h5ad | signatures.csv}. Without
links the same files sit side by side anywhere under INPUT_DIR (symlinks followed): then an
.h5ad with obsm["spatial"] is a spatial sample and one without is the reference.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import pandas as pd

log = logging.getLogger("cell2location-app")


@dataclass
class Inputs:
    spatial: list[tuple[str, Path]]           # (sample_id, path), sorted by sample id
    reference: Path | None                    # .h5ad or signatures.csv
    reference_kind: str | None                # "h5ad" or "signatures"
    samples_csv: Path | None
    samples: pd.DataFrame | None
    skipped: dict[str, str] = field(default_factory=dict)


def walk(root: Path) -> list[Path]:
    """Every file under root, following symlinks (INPUT_DIR entries are symlinks), sorted."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        found.extend(Path(dirpath) / f for f in sorted(filenames) if not f.startswith("."))
    return found


def _find(root: Path, port: str, predicate) -> tuple[list[Path], bool]:
    """Files of a port: its own folder when it exists (True), else anywhere under INPUT_DIR (False)."""
    folder = root / port
    if folder.is_dir():
        return [p for p in walk(folder) if predicate(p)], True
    return [p for p in walk(root) if predicate(p)], False


def has_spatial_coords(path: Path) -> bool | None:
    """Whether an .h5ad has obsm["spatial"] (read from the file's structure, without loading it)."""
    try:
        with h5py.File(path, "r") as f:
            return "obsm" in f and "spatial" in f["obsm"]
    except OSError:
        return None


def read_samples_csv(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "sample_id" not in table.columns:
        raise ValueError(f"{path.name}: no sample_id column")
    return table


def discover(input_dir: Path) -> Inputs:
    is_h5ad = lambda p: p.suffix.lower() == ".h5ad"  # noqa: E731
    skipped: dict[str, str] = {}

    csvs, _ = _find(input_dir, "metadata", lambda p: p.name.lower() == "samples.csv")
    samples_csv = csvs[0] if csvs else None
    if len(csvs) > 1:
        log.warning("%d samples.csv files; using %s", len(csvs), csvs[0])
    samples = read_samples_csv(samples_csv) if samples_csv else None

    spatial_files, spatial_linked = _find(input_dir, "spatial", is_h5ad)
    ref_folder = input_dir / "reference"
    if ref_folder.is_dir():
        ref_files = [p for p in walk(ref_folder) if is_h5ad(p) or p.name.lower() == "signatures.csv"]
    else:
        # No reference folder: a signatures.csv anywhere, else an .h5ad without spatial coordinates.
        ref_files = [p for p in walk(input_dir) if p.name.lower() == "signatures.csv"]
        if not ref_files:
            ref_files = [p for p in spatial_files if has_spatial_coords(p) is False]
    if not spatial_linked:
        spatial_files = [p for p in spatial_files if p not in ref_files]

    reference, kind = None, None
    sigs = [p for p in ref_files if p.name.lower() == "signatures.csv"]
    h5ads = [p for p in ref_files if is_h5ad(p)]
    if sigs:
        reference, kind = sigs[0], "signatures"
        if len(sigs) > 1 or h5ads:
            log.warning("several reference files; using %s", reference)
    elif h5ads:
        reference, kind = h5ads[0], "h5ad"
        if len(h5ads) > 1:
            log.warning("%d reference .h5ad files; using %s", len(h5ads), reference)

    by_stem: dict[str, Path] = {}
    for p in spatial_files:
        if p.stem in by_stem:
            skipped[str(p)] = f"two files for sample {p.stem}; using {by_stem[p.stem]}"
            continue
        by_stem[p.stem] = p
    if samples is not None:
        listed = set(samples["sample_id"])
        for sid in listed - set(by_stem):
            skipped[sid] = "listed in samples.csv but no .h5ad found"
        for stem in set(by_stem) - listed:
            log.warning("%s.h5ad is not listed in samples.csv; using it anyway", stem)
    spatial = sorted(by_stem.items())
    return Inputs(spatial, reference, kind, samples_csv, samples, skipped)
