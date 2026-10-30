"""
Gene identity: symbols and Ensembl ids on both sides, and matching the signatures to the
spatial genes.

The reference (and so signatures.csv) carries a symbol per gene and, when it has one, an
Ensembl id; the spatial contract has symbols as var_names and the platform's ids in
var["gene_ids"]. Genes are matched by Ensembl id where both sides have one, then by symbol
(case-sensitive), then by symbol ignoring case; each signature gene is used once and each
spatial gene at most once.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{5,}(\.\d+)?$")
ID_COLUMNS = ("gene_ids", "gene_id", "ensembl_id", "ensembl", "feature_id")
SYMBOL_COLUMNS = ("feature_name", "gene_symbols", "gene_symbol", "symbol", "gene_name", "SYMBOL")


def looks_ensembl(values) -> bool:
    v = [str(x) for x in list(values)[:200]]
    return bool(v) and sum(bool(ENSEMBL_RE.match(x)) for x in v) >= 0.8 * len(v)


def strip_version(ids: pd.Series) -> pd.Series:
    return ids.astype(str).str.replace(r"\.\d+$", "", regex=True)


def symbols_and_ids(var: pd.DataFrame) -> tuple[pd.Series, pd.Series | None]:
    """(symbol per gene, Ensembl id per gene or None), both indexed like var."""
    names = pd.Series(var.index.astype(str), index=var.index)
    ids = None
    if looks_ensembl(names):
        ids = strip_version(names)
    else:
        for c in ID_COLUMNS:
            if c in var.columns and looks_ensembl(var[c].astype(str)):
                ids = strip_version(var[c])
                break
    symbols = None
    for c in SYMBOL_COLUMNS:
        if c in var.columns:
            s = var[c].astype(str)
            if not looks_ensembl(s):
                symbols = s
                break
    if symbols is None:
        symbols = names
    symbols = symbols.where(symbols.str.len() > 0, names)
    symbols = symbols.where(symbols.str.lower() != "nan", names)
    return symbols, ids


def match_genes(spatial_var: pd.DataFrame, signatures: pd.DataFrame) -> pd.DataFrame:
    """
    Rows (signature gene -> spatial gene), one per matched signature gene, in signature order:
    columns gene (signature symbol), gene_id, spatial_gene (spatial var_name), match
    (ensembl / symbol / symbol_ci). signatures: index = symbol, optional column gene_id.
    """
    sp_symbol, sp_ids = symbols_and_ids(spatial_var)
    sp_names = pd.Series(spatial_var.index.astype(str), index=spatial_var.index)
    sig_symbol = pd.Series(signatures.index.astype(str), index=signatures.index)
    sig_ids = strip_version(signatures["gene_id"]) if "gene_id" in signatures.columns else None

    used_spatial: set[str] = set()
    out = {}   # signature index -> (spatial var_name, how)

    def take(mapping: dict[str, str], keys: pd.Series, how: str):
        for idx, key in keys.items():
            if idx in out or not key or key == "nan":
                continue
            name = mapping.get(key)
            if name is not None and name not in used_spatial:
                out[idx] = (name, how)
                used_spatial.add(name)

    def unique_map(keys: pd.Series) -> dict[str, str]:
        counts = keys.value_counts()
        return {k: n for k, n in zip(keys.values, sp_names.values) if counts[k] == 1}

    if sp_ids is not None and sig_ids is not None:
        take(unique_map(sp_ids), sig_ids, "ensembl")
    take(unique_map(sp_names), sig_symbol, "symbol")
    if not sp_symbol.equals(sp_names):
        take(unique_map(sp_symbol), sig_symbol, "symbol")
    take(unique_map(sp_names.str.upper()), sig_symbol.str.upper(), "symbol_ci")
    if not sp_symbol.equals(sp_names):
        take(unique_map(sp_symbol.str.upper()), sig_symbol.str.upper(), "symbol_ci")

    rows = []
    for idx in signatures.index:
        if idx in out:
            name, how = out[idx]
            rows.append({"gene": str(idx), "gene_id": "" if sig_ids is None else sig_ids[idx],
                         "spatial_gene": name, "match": how})
    return pd.DataFrame(rows, columns=["gene", "gene_id", "spatial_gene", "match"])


def match_counts(genes_used: pd.DataFrame, n_signature: int, n_spatial: int) -> dict:
    by = genes_used["match"].value_counts().to_dict() if len(genes_used) else {}
    return {"n_signature_genes": int(n_signature), "n_spatial_genes": int(n_spatial),
            "n_matched": int(len(genes_used)),
            "by_ensembl_id": int(by.get("ensembl", 0)), "by_symbol": int(by.get("symbol", 0)),
            "by_symbol_case_insensitive": int(by.get("symbol_ci", 0)),
            "fraction_of_signature_genes": round(len(genes_used) / n_signature, 4) if n_signature else 0.0}


def make_unique(values: pd.Series) -> pd.Index:
    """anndata-style unique names: duplicates get -1, -2, ..."""
    import anndata.utils as au

    return au.make_index_unique(pd.Index(values.astype(str)), join="-")


def read_signatures(path) -> pd.DataFrame:
    """signatures.csv: index = gene symbol, optional gene_id column, one numeric column per cell type."""
    sig = pd.read_csv(path, index_col=0)
    sig.index = sig.index.astype(str)
    if "gene_id" in sig.columns:
        sig["gene_id"] = sig["gene_id"].fillna("").astype(str)
    types = [c for c in sig.columns if c != "gene_id"]
    if not types:
        raise ValueError(f"{path}: no cell type columns")
    bad = [c for c in types if not np.issubdtype(sig[c].dtype, np.number)]
    if bad:
        raise ValueError(f"{path}: non-numeric columns {bad[:5]}")
    if sig.index.duplicated().any():
        sig.index = make_unique(pd.Series(sig.index))
    return sig


def signature_types(sig: pd.DataFrame) -> list[str]:
    return [c for c in sig.columns if c != "gene_id"]
