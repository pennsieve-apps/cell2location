"""The synthetic fixture through the processor (no Docker), with few epochs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from processor import genes as G
from processor import main as P
from processor import mapping as M
from processor import resources as RS
from tests import make_fixture as FX

FAST = {"REFERENCE_EPOCHS": "40", "SPATIAL_EPOCHS": "1500", "THREADS": "2", "USE_GPU": "false",
        "REFERENCE_BATCH_COLUMN": "donor"}


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("fixture")
    for layout in ("linked", "flat"):
        FX.write(root / layout, layout, seed=7)
    return root


def run(input_dir: Path, output_dir: Path, monkeypatch, **params) -> int:
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    for k in P.PARAMETERS:
        monkeypatch.delenv(k, raising=False)
    for k, v in {**FAST, **params}.items():
        monkeypatch.setenv(k, str(v))
    return P.main()


@pytest.fixture(scope="module")
def linked_run(fixture_root, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("out-linked")
    with pytest.MonkeyPatch.context() as mp:
        assert run(fixture_root / "linked", out, mp) == 0
    return out


def test_outputs_follow_the_contract(linked_run, fixture_root):
    labels, model = linked_run / "labels", linked_run / "model"
    for name in ("labels.csv", "abundance.csv", "proportions.csv", "colocation.csv", "colocation_by_obs.csv",
                 "type_summary.csv", "summary.json", "provenance.json"):
        assert (labels / name).is_file(), name
    for name in ("signatures.csv", "reference_summary.csv", "reference_history.csv", "spatial_history.csv",
                 "genes_used.csv", "provenance.json"):
        assert (model / name).is_file(), name
    assert not [p for p in model.rglob("*") if p.suffix in (".pt", ".pkl", ".h5ad")]   # no model objects

    lab = pd.read_csv(labels / "labels.csv")
    assert list(lab.columns) == ["sample_id", "obs_id", "dominant_type", "colocation_group"]
    assert lab["colocation_group"].str.match(r"^G\d\d$").all()   # every spot gets a group
    n = sum(ad.read_h5ad(fixture_root / "linked" / "spatial" / f"{s}.h5ad").n_obs for s in FX.SAMPLES)
    assert len(lab) == n and lab["obs_id"].is_unique
    assert set(lab["sample_id"]) == set(FX.SAMPLES)
    assert set(lab["dominant_type"]) <= set(FX.TYPES) | {"Mixed"}

    ab = pd.read_csv(labels / "abundance.csv")
    assert list(ab.columns[:2]) == ["sample_id", "obs_id"] and sorted(ab.columns[2:]) == sorted(FX.TYPES)
    assert (ab[FX.TYPES] >= 0).all().all()
    pr = pd.read_csv(labels / "proportions.csv")
    assert np.allclose(pr[FX.TYPES].sum(axis=1), 1.0, atol=1e-4)
    co = pd.read_csv(labels / "colocation.csv")
    assert list(co.columns) == ["group", "cell_type", "weight"]
    assert np.allclose(co.groupby("cell_type")["weight"].sum(), 1.0, atol=1e-4)
    cbo = pd.read_csv(labels / "colocation_by_obs.csv")
    assert len(cbo) == n and set(cbo.columns[2:]) == set(co["group"])

    sig = pd.read_csv(model / "signatures.csv", index_col=0)
    assert "gene_id" in sig.columns and sorted(c for c in sig.columns if c != "gene_id") == sorted(FX.TYPES)
    rs = pd.read_csv(model / "reference_summary.csv")
    assert rs.set_index("cell_type").loc["Rare", "status"].startswith("dropped")

    summary = json.loads((labels / "summary.json").read_text())
    assert summary["n_ok"] == 2 and summary["n_cell_types"] == 4
    assert summary["genes"]["n_matched"] == len(sig)
    assert summary["colocation"]["n_groups"] >= 2 and summary["colocation"]["reason"]
    prov = json.loads((model / "provenance.json").read_text())
    assert prov["app"] == "cell2location" and prov["status"] == "ok" and prov["tools"]["cell2location"]
    assert prov["device"]["accelerator"] == "cpu"
    figs = {p.name for p in (labels / "figures").iterdir()}
    for sid in FX.SAMPLES:
        assert {f"{sid}_abundance.png", f"{sid}_dominant_type.png", f"{sid}_colocation.png"} <= figs
    assert {"training_history.png", "reconstruction_qc.png", "colocation_weights.png"} <= figs


def test_planted_structure_is_recovered(linked_run, fixture_root):
    ab = pd.read_csv(linked_run / "labels" / "abundance.csv", index_col="obs_id")
    lab = pd.read_csv(linked_run / "labels" / "labels.csv", index_col="obs_id")["dominant_type"]
    for sid in FX.SAMPLES:
        a = ad.read_h5ad(fixture_root / "linked" / "spatial" / f"{sid}.h5ad")
        truth = pd.DataFrame(a.obsm["planted_abundance"], index=a.obs_names, columns=FX.TYPES)
        est = ab.loc[a.obs_names]
        for t in ("B_cell", "T_cell", "Myeloid"):
            assert np.corrcoef(truth[t], est[t])[0, 1] > 0.8, (sid, t)
        follicle = a.obs["planted_follicle"].to_numpy()
        assert (lab.loc[a.obs_names][follicle] == "B_cell").mean() > 0.85
        assert (lab.loc[a.obs_names][~follicle] == "T_cell").mean() > 0.5


def test_flat_layout_with_signatures(linked_run, fixture_root, tmp_path, monkeypatch):
    """Without links; a signatures.csv from an earlier run replaces the reference .h5ad (stage 1 skipped)."""
    flat = tmp_path / "flat"
    shutil.copytree(fixture_root / "flat", flat)
    (flat / "reference.h5ad").unlink()
    shutil.copy(linked_run / "model" / "signatures.csv", flat / "signatures.csv")
    out = tmp_path / "out"
    assert run(flat, out, monkeypatch, SPATIAL_EPOCHS=300, N_COLOCATION_GROUPS=2, MAX_PLOTS=0) == 0
    prov = json.loads((out / "model" / "provenance.json").read_text())
    assert prov["inputs"]["reference_kind"] == "signatures"
    assert not (out / "model" / "reference_history.csv").exists()
    assert (out / "model" / "signatures.csv").read_text() == (linked_run / "model" / "signatures.csv").read_text()
    summary = json.loads((out / "labels" / "summary.json").read_text())
    assert summary["colocation"]["n_groups"] == 2 and summary["n_obs"] == 2 * FX.GRID ** 2
    assert not list((out / "labels" / "figures").glob("*_abundance.png"))


def test_a_bad_sample_is_skipped(fixture_root, tmp_path, monkeypatch):
    src = tmp_path / "in"
    shutil.copytree(fixture_root / "linked", src)
    (src / "spatial" / "lymph_b.h5ad").unlink()
    (src / "spatial" / "broken.h5ad").write_bytes(b"not an h5ad")
    out = tmp_path / "out"
    assert run(src, out, monkeypatch, REFERENCE_EPOCHS=5, SPATIAL_EPOCHS=50, MAX_PLOTS=1) == 0
    prov = json.loads((out / "labels" / "provenance.json").read_text())
    status = {s["sample_id"]: s["status"] for s in prov["samples"]}
    assert status == {"broken": "failed", "lymph_a": "ok"}
    assert set(pd.read_csv(out / "labels" / "labels.csv")["sample_id"]) == {"lymph_a"}


def test_no_reference_fails(fixture_root, tmp_path, monkeypatch):
    src = tmp_path / "in"
    shutil.copytree(fixture_root / "linked", src)
    shutil.rmtree(src / "reference")
    assert run(src, tmp_path / "out", monkeypatch) == 1


def test_wrong_cell_type_column_fails(fixture_root, tmp_path, monkeypatch):
    assert run(fixture_root / "linked", tmp_path / "out", monkeypatch, CELL_TYPE_COLUMN="celltype") == 1


@pytest.mark.parametrize("params", [{"MIN_DOMINANT_FRACTION": "1.5"}, {"SPATIAL_EPOCHS": "0"},
                                    {"USE_GPU": "maybe"}, {"N_COLOCATION_GROUPS": "1"}, {"SEED": "x"},
                                    {"MAX_CELLS_PER_TYPE": "5"}])
def test_bad_parameters(fixture_root, tmp_path, monkeypatch, params):
    assert run(fixture_root / "linked", tmp_path / "out", monkeypatch, **params) == 2


def test_gene_matching_by_symbol_and_case():
    sig = pd.DataFrame({"A": [1.0, 2, 3, 4], "B": [4.0, 3, 2, 1]}, index=["CD19", "Ms4a1", "cd3e", "XIST"])
    var = pd.DataFrame(index=["CD19", "MS4A1", "CD3E", "GAPDH"])
    used = G.match_genes(var, sig)
    assert dict(zip(used["gene"], used["match"])) == {"CD19": "symbol", "Ms4a1": "symbol_ci", "cd3e": "symbol_ci"}
    assert list(used["spatial_gene"]) == ["CD19", "MS4A1", "CD3E"]
    counts = G.match_counts(used, len(sig), len(var))
    assert counts["n_matched"] == 3 and counts["by_symbol_case_insensitive"] == 2


def test_gene_matching_prefers_ensembl_ids():
    sig = pd.DataFrame({"gene_id": ["ENSG00000177455", "ENSG00000156738"], "A": [1.0, 2.0]}, index=["CD19", "OLDNAME"])
    var = pd.DataFrame({"gene_ids": ["ENSG00000156738", "ENSG00000177455"]}, index=["MS4A1", "CD19"])
    used = G.match_genes(var, sig)
    assert dict(zip(used["gene"], used["spatial_gene"])) == {"CD19": "CD19", "OLDNAME": "MS4A1"}
    assert set(used["match"]) == {"ensembl"}


def test_dominant_types():
    q = pd.DataFrame({"A": [5.0, 1.0, 0.0], "B": [1.0, 1.0, 0.0], "C": [0.0, 1.2, 0.0]})
    label, frac = M.dominant_types(q, 0.4)
    assert list(label) == ["A", "Mixed", "Mixed"]
    assert frac.iloc[0] == pytest.approx(5 / 6)


def test_cgroup_cpus(tmp_path):
    (tmp_path / "cpu.max").write_text("250000 100000\n")
    assert RS.cgroup_cpus(tmp_path) == 2.5
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert RS.cgroup_cpus(tmp_path) is None
    assert RS.choose_device("false")["accelerator"] == "cpu"
