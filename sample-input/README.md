# sample-input

This app reads three inputs (the contract is in
`workflow-spatial-transcriptomics/docs/conventions.md`):

- `spatial/<sample_id>.h5ad`: the output of `st-qc` (raw counts in `layers["counts"]`) or of
  `st-prep` (raw counts in `X`), with `obsm["spatial"]`.
- `metadata/samples.csv`: written by `st-prep`.
- `reference/`: a single-cell RNA-seq `.h5ad` of the same tissue and species with **raw counts**
  and a cell type column, or a `signatures.csv` from an earlier run of this app (its `model`
  output).

## Synthetic fixture (no download)

```sh
make fixture PYTHON=.venv/bin/python   # data/fixture/{linked,flat}/
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked/ -> data/output/{labels,model}/
```

A reference of 486 cells (B_cell, T_cell, Stromal, Myeloid at 120 cells each, and Rare with 6
cells, below `MIN_CELLS_PER_TYPE`; two donors), 160 genes with Ensembl-style `var_names` and
symbols in `var["feature_name"]`; two st-qc-style Visium samples of 196 spots simulated from the
same signatures, with B cells in two round "follicles", T cells around them, stroma everywhere
and myeloid cells in one corner (`obs["planted_follicle"]`, `obsm["planted_abundance"]` hold the
truth).

## Real data: Human Lymph Node (the cell2location tutorial's dataset)

From the workflow repository (`scripts/fetch-sample-data.sh`):

- `data/lymph-node/V1_Human_Lymph_Node/`: 10x Genomics *Human Lymph Node* (Visium, Space Ranger
  1.1.0, CC BY 4.0), 4,035 spots. Run `st-prep` then `st-qc` on it.
- `data/lymph-node/reference/tabula-sapiens-lymph-node.h5ad`: the Tabula Sapiens lymph node
  data (The Tabula Sapiens Consortium, Science 2022, doi:10.1126/science.abl4896; CELLxGENE
  collection e5f58829-1a66-40b5-a624-9046778e74f5, CC BY 4.0), subsampled to at most 500 cells
  per type: 7,296 cells, 23 cell types, raw counts in `X`, Ensembl ids as `var_names` and
  symbols in `var["feature_name"]`, `obs["donor_id"]`.

Any CELLxGENE `.h5ad` works as a reference: the raw counts are in `.raw.X` (or `X`), the cell
type in `obs["cell_type"]`, symbols in `var["feature_name"]`. Keep it to the same tissue and
include every cell type you expect in the sections: a type missing from the reference cannot be
found, and its RNA is attributed to the most similar types present.

Then (input folders as the platform would link them):

```sh
mkdir -p in && ln -s /path/to/st-qc/output/spatial in/spatial \
  && ln -s /path/to/st-prep/output/metadata in/metadata && ln -s /path/to/reference-folder in/reference
docker run --rm --platform linux/amd64 --memory 4g --cpus 4 --network none \
  -e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output -e REFERENCE_BATCH_COLUMN=donor_id \
  -e SPATIAL_EPOCHS=5000 \
  -v "$PWD/in:/data/input:ro" -v "$PWD/data/output:/data/output" cell2location:dev
```

(Docker does not follow symlinks into folders outside the mount; on a laptop copy the three
folders into `in/` instead.)
