# cell2location

Cell-type deconvolution of spot-based spatial transcriptomics with a single-cell reference, the
step after `st-qc` in the **Cell-type Deconvolution & Niches** workflow:

- **B. Cell-type Deconvolution & Niches:** `st-prep → st-qc → cell2location (+ reference input) → st-niches → st-report`

A 55 µm Visium spot holds somewhere between a few and a few dozen cells, so its RNA is a mixture.
Clustering spots finds tissue regions, but cannot tell you *which cells* make up a region or how
many of each there are. This app answers that with [cell2location](https://github.com/BayraktarLab/cell2location)
(Kleshchevnikov et al. 2022): given a single-cell RNA-seq reference of the same tissue with
annotated cell types, it estimates for every spot the **absolute number of cells of each type**
(not just proportions), together with how sure it is. From those abundances it labels each spot
with its dominant cell type (or `Mixed`), and finds **groups of cell types that tend to occur
together** (tissue zones or micro-environments, such as a germinal centre: B cells, follicular
dendritic cells and T follicular helper cells).

For the computational analyst: the app runs the three stages of the official
[lymph node tutorial](https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html):
(1) `RegressionModel` (a negative-binomial regression that removes batch effects of the
reference) gives a reference expression signature per cell type, after the tutorial's
`filter_genes`; (2) `Cell2location`, a hierarchical Bayesian model of the spatial counts as a
sum of those signatures, trained jointly on all samples (`batch_key="sample_id"`: a detection
efficiency per sample) with full-data variational inference; the q05 of the posterior cell
abundance is exported; (3) `CoLocatedGroupsSklearnNMF` (what `run_colocation` fits) on the
abundances. All of this runs on CPU or, when the step has a GPU, on the GPU.

## Ports

| Port | Direction | Content |
|---|---|---|
| `spatial` | input (subfolder, required) | `<sample_id>.h5ad` from `st-qc` (raw counts in `layers["counts"]`) or `st-prep` (raw counts in `X`), with `obsm["spatial"]`; spot-based data (Visium). |
| `metadata` | input (subfolder, required) | `st-prep`'s `samples.csv`. |
| `reference` | input (subfolder, required) | A scRNA-seq `.h5ad` with raw counts (`layers["counts"]`, `X` or `.raw.X`, whichever is integer-valued) and a cell type column; or a `signatures.csv` from an earlier run's `model` output (skips stage 1). |
| `labels` | output | `labels.csv`, `abundance.csv`, `proportions.csv`, `colocation.csv`, `colocation_by_obs.csv`, `type_summary.csv`, `summary.json`, `figures/`, `provenance.json` |
| `model` | output | `signatures.csv`, `reference_summary.csv`, `reference_history.csv`, `spatial_history.csv`, `genes_used.csv`, `provenance.json` (no model objects) |

Without links everything may sit side by side anywhere under `INPUT_DIR`: then `.h5ad` files
with `obsm["spatial"]` are samples, one without is the reference, and a `signatures.csv` anywhere
is taken as the reference. The step is single-run: it writes at the root of each output port.

## Method

1. **Reference signatures** (skipped when the reference is a `signatures.csv`). Raw counts are
   taken from `layers["counts"]`, `X` or `.raw.X` (the first that is integer-valued); gene symbols
   from `var["feature_name"]` (CELLxGENE), `gene_symbols`, ... or `var_names`, Ensembl ids from
   `var_names` or `var["gene_ids"]`. Cell types with fewer than `MIN_CELLS_PER_TYPE` cells are
   dropped (a signature from a handful of cells is noise), larger ones are subsampled (seeded) to
   `MAX_CELLS_PER_TYPE`. Genes are selected with cell2location's `filter_genes` (the tutorial's
   thresholds: drop genes in fewer than `CELL_COUNT_CUTOFF` cells, keep those in more than
   `CELL_PERCENTAGE_CUTOFF2` of cells, and in between those whose non-zero mean exceeds
   `NONZ_MEAN_CUTOFF`). `RegressionModel` (`batch_key` = `REFERENCE_BATCH_COLUMN`, e.g. donor) is
   trained for `REFERENCE_EPOCHS` epochs (batch size 2500, lr 0.002, as in the tutorial); the
   posterior mean expression per cell type (`means_per_cluster_mu_fg`, 1000 posterior samples) is
   the signature.
2. **Spatial mapping.** Every sample's raw counts are concatenated; mitochondrially encoded
   genes are left out (as in the tutorial); genes are matched to the signatures by Ensembl id
   where both sides have one, then by symbol, then by symbol ignoring case, and the overlap is
   reported (`genes_used.csv`, `summary.json` → `genes`). `Cell2location(N_cells_per_location,
   detection_alpha)` is trained for `SPATIAL_EPOCHS` epochs on all spots at once (`batch_size=None`,
   `train_size=1`), and the 5 % quantile of the posterior cell abundance (`q05`, "the number of
   cells we are confident are present") is exported (1000 posterior samples).
3. **Dominant type** per spot: the type with the largest q05 abundance, `Mixed` when it holds
   less than `MIN_DOMINANT_FRACTION` of the spot's estimated cells.
4. **Co-location groups**: `CoLocatedGroupsSklearnNMF` (the model `run_colocation` fits, with its
   defaults: random init, alpha 0.01, l1 ratio 0.5, 20000 iterations, 5 restarts) on the q05
   abundances of all spots. With `N_COLOCATION_GROUPS=0` it tries 4 to 8 groups (fewer than the
   number of cell types) and keeps the **largest number whose cell-type loadings are stable across
   the 5 NMF restarts** (mean correlation of the aligned loadings ≥ 0.9, the diagnostic
   cell2location plots with `evaluate_stability`), else the most stable one; the candidates and the
   reason are in `summary.json` → `colocation`. Groups are named `G01`, `G02`, ... by decreasing
   total loading.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `CELL_TYPE_COLUMN` | `cell_type` | Column of the reference's `obs` with the cell type. |
| `REFERENCE_BATCH_COLUMN` | `""` | Optional `obs` column with the reference batch (donor, sample, chemistry); its effect is regressed out of the signatures. Empty: one batch. |
| `MIN_CELLS_PER_TYPE` | `20` | Cell types with fewer reference cells are dropped (listed in `reference_summary.csv`). |
| `MAX_CELLS_PER_TYPE` | `2000` | Reference cells per type subsampled (seeded) to at most this; `0` keeps all. |
| `CELL_COUNT_CUTOFF` | `5` | `filter_genes`: genes in fewer reference cells are always dropped. |
| `CELL_PERCENTAGE_CUTOFF2` | `0.03` | `filter_genes`: genes in at least this fraction of cells are always kept. |
| `NONZ_MEAN_CUTOFF` | `1.12` | `filter_genes`: genes in between are kept when their mean in expressing cells is above this. |
| `REFERENCE_EPOCHS` | `250` | Epochs of the reference regression model. |
| `N_CELLS_PER_LOCATION` | `8` | Prior expected number of cells per spot; best estimated by counting nuclei in a few spots of the H&E. The lymph node tutorial uses 30 (a dense lymphoid tissue); the cell2location docs suggest 5-30 for Visium depending on the tissue. |
| `DETECTION_ALPHA` | `20` | Regularisation of the per-spot detection efficiency: 20 when it varies within a slide (most data), 200 for very even data. |
| `SPATIAL_EPOCHS` | `5000` (cpu), `30000` (gpu) | Epochs of the spatial model; 30000 is the tutorial's value. See Variants. |
| `MIN_DOMINANT_FRACTION` | `0.4` | `dominant_type` is `Mixed` below this share of the spot's cells. |
| `N_COLOCATION_GROUPS` | `0` | NMF groups; `0` chooses among 4-8 (see Method). |
| `MAX_PLOTS` | `8` | Abundance maps for the this-many most abundant types per sample; `0` none. |
| `SEED` | `42` | Seed for subsampling, initialisation, training and posterior sampling. |
| `THREADS` | `0` | CPU threads (torch, BLAS); `0` = the container's CPU quota. |
| `USE_GPU` | `false` (cpu), `auto` (gpu) | `auto`: the GPU when torch sees one; `true`: same, with a warning when there is none; `false`: CPU. |

Exit codes: `0` ok, `1` no usable input (no reference, no readable sample, reference or mapping
failed, fewer than 50 shared genes), `2` bad parameters. A sample that cannot be read is logged
and skipped.

## Outputs

`labels/`

- `labels.csv`: `sample_id, obs_id, dominant_type, colocation_group` (every spot of every sample).
  `colocation_group` is the spot's strongest co-location group (`G01`, ...; the largest loading in
  `colocation_by_obs.csv`), present when the co-location NMF ran. Every spot has one, so it is the
  label to use for niche analysis (st-niches `LABEL_COLUMN=colocation_group`); `dominant_type` is
  mostly `Mixed` on a fine-grained reference (below).
- `abundance.csv`: `sample_id, obs_id`, one column per cell type: q05 posterior number of cells.
- `proportions.csv`: the same divided by each spot's total.
- `colocation.csv`: `group, cell_type, weight` (the fraction of that cell type's abundance
  explained by the group; sums to 1 per cell type).
- `colocation_by_obs.csv`: `sample_id, obs_id`, one column per group (NMF location loading, in
  cells).
- `type_summary.csv`: `sample_id, cell_type, total_abundance, mean_abundance, fraction_of_cells,
  n_dominant, fraction_dominant`.
- `summary.json`: reference (cells and genes used, dropped types, final ELBO, reconstruction
  correlation), gene overlap, mapping (epochs, final ELBO, reconstruction correlation, mean cells
  per spot, device), dominant type counts, co-location (candidates, chosen number and why, top
  types per group), per-sample numbers.
- `figures/`: `<sample_id>_abundance.png` (q05 maps of the `MAX_PLOTS` most abundant types, on the
  H&E), `<sample_id>_dominant_type.png`, `<sample_id>_colocation.png` (group loadings),
  `training_history.png` (ELBO of both models), `reconstruction_qc.png` (observed vs expected
  counts of both models and signatures vs plain cluster means, the panels of cell2location's
  `plot_QC`), `reference_gene_filter.png`, `colocation_weights.png`, `type_composition.png`.
- `provenance.json`: versions, parameters, device, inputs, method, citations, timings, peak memory.

`model/`

- `signatures.csv`: index = gene symbol, `gene_id` (Ensembl, when known), one column per cell type
  (expected counts per cell). Link it as the `reference` of a later run to skip stage 1.
- `reference_summary.csv`: `cell_type, n_cells, n_used, status` (used / dropped and why).
- `reference_history.csv`, `spatial_history.csv`: `epoch, elbo`.
- `genes_used.csv`: `gene, gene_id, spatial_gene, match` (`ensembl`, `symbol`, `symbol_ci`).
- `provenance.json`.

## Runtime

Measured on the lymph node demo (one Visium section after `st-qc`: 3,928 spots; reference 7,269
cells of 20 types; 24,117 reference genes after `filter_genes`, 17,375 shared with the spatial data)
on an M-series Mac shared with other builds, so wall times carry some contention:

| What | Native arm64 (venv) | Docker, linux/amd64 emulation, `--cpus 4` |
|---|---|---|
| Reference model, 250 epochs | 3.2 s/epoch on 4 threads (~13 min); 405-450 s on 10 threads | 4.6 s/epoch with 9,245 genes (stricter `filter_genes`), 1,152 s |
| Spatial model, per epoch (17,375 genes) | 1.24 s on 4 threads, 0.63 s on 10 threads | 4.2 s |
| Spatial model, per epoch (8,685 genes) | - | 2.2 s |
| Peak memory | 10.1 GB max RSS for the whole run (reference model on 24,117 genes + spatial model on 17,375 genes); 6.3 GB for the spatial stage alone | > 6 GB (OOM-killed at `--memory 6g` during training with 17,375 genes; at `--memory 4g` after 2,000 epochs with 8,685 genes, during posterior export) |

Memory grows with spots × genes (about 80 bytes per entry on top of ~1 GB), so the app asks
for 4 vCPU / 16 GiB. The tutorial's 30000 spatial epochs on CPU would take about 10.5 h on 4
threads (1.24 s × 30000, native arm64; slower in amd64 emulation, ~35 h); we did not run it to
the end here (two attempts were stopped when the machine was shared, after the reference stage).
That is why the variants below default to 5000 epochs on CPU and 30000 on a GPU, where the
cell2location authors report the full training in minutes to tens of minutes per section.

**The demo output** (`workflow-spatial-transcriptomics/data/work/lymph/deconvolution/`) is from
`REFERENCE_BATCH_COLUMN=donor_id`, `SPATIAL_EPOCHS=2000`, everything else default, run
natively on 10 threads (49 min: reference 9 min, spatial 2000 epochs 33 min, co-location 43 s; max RSS 10.1 GB; finished 10:26 EDT, 2026-10-30). See "Demo results" below.

The image runs as uid 1000 with no network; `--memory` must be at least 8 GB for a Visium section
with the default gene filter.

## Variants

One release builds two images (`app.yml` → `variants`, Dockerfile `ARG TORCH_FLAVOUR`):

| Variant | Image | Compute | Defaults | Pick it when |
|---|---|---|---|---|
| `cpu` (default, "CPU") | CPU-only PyTorch 2.14.1+cpu from the PyTorch CPU index, no NVIDIA libraries: 0.48 GB compressed, ~2.1 GB unpacked | `standard`, 4 vCPU / 16 GiB, 8 h timeout | `SPATIAL_EPOCHS=5000`, `USE_GPU=false` | No GPU is available or a 2-hour run is fine. ~2 h for one section with the default genes (5000 × 1.24 s + reference). |
| `gpu` ("GPU") | CUDA 12.6 PyTorch 2.14.1 (NVIDIA libraries as pip wheels; runs on the platform's driver 550, T4 or newer): 4.0 GB compressed, ~11.4 GB unpacked | `gpu`, 14 GiB, 4 h timeout | `SPATIAL_EPOCHS=30000` (the tutorial's), `USE_GPU=auto` | A GPU is available: the full tutorial setting. |

The GPU speed-up is **expected, not measured** here (no GPU on the build machine): the model is
dense matrix algebra over spots × genes, and on a T4 we expect one to two orders of magnitude per
epoch, i.e. the 30000 epochs in well under an hour for a Visium section. The cuda image was
checked to start and fall back cleanly to CPU without a GPU (`USE_GPU=auto: cuda not available,
using cpu`), and the cpu image ran the fixture in Docker as uid 1000 (316 s, cgroup peak 0.99 GB).

**Quality trade-off of fewer epochs** (measured on the demo; no 30000-epoch CPU run was
available to compare with): going from 1000 to 2000 epochs changed the abundances only a little
(per-type correlation across spots median 0.95, lowest 0.75 for the small generic "T cell" type;
mean |difference| of the per-spot proportions 0.009; dominant type unchanged in 98.7 % of spots;
cell totals within 2 %) but sharpened the tissue structure: B cells in the top-10 % follicle-marker
spots went from 1.3 to 2.3 cells per spot and the B/T anticorrelation across spots from -0.11 to
-0.30. The loss is still falling at 2000 epochs (`training_history.png`), so 5000 epochs (the cpu
default) is a compromise: the spatial pattern and the ranking of types are stable early, while the
contrast between zones keeps improving towards the tutorial's 30000. Use the gpu variant when
absolute abundances matter.

## Demo results

Human Lymph Node (10x Visium, 3,928 spots after st-qc) with the Tabula Sapiens lymph node
reference (20 of 23 cell types kept; hematopoietic precursor cell, innate lymphoid cell and
plasmacytoid dendritic cell have fewer than 20 cells), 2000 spatial epochs:

- Genes: 24,117 reference genes pass `filter_genes`; 17,375 are shared with the section
  (17,366 by Ensembl id, 9 by symbol; 13 mitochondrial genes left out). Reconstruction
  correlation (log1p observed vs expected counts) 0.72 for the reference model and 0.82 for the
  spatial model.
- Mean 6.9 estimated cells per spot (q05). Largest shares: B cell 16 %, regulatory T cell 9 %,
  plasma cell 9 %, erythrocyte 7 %, naive CD4 T cell 7 %, endothelial cell 6.5 %.
- **Sanity check: B cells sit in the follicles, T cells in the paracortex.** The B cell abundance
  correlates with the follicle markers MS4A1/CR2/CD19/FCER2/CXCL13 (r = 0.78), the summed T cell
  abundance with the T-zone markers CCR7/CD3E/CD3D/LEF1/CCL19/CCL21 (r = 0.78), and B and T
  abundances are anticorrelated (r = -0.30). In the 10 % of spots richest in follicle markers there
  are 2.3 B vs 1.8 T cells per spot; in the 10 % richest in T-zone markers 0.4 B vs 5.0 T cells.
- Co-location: 8 groups (4-8 all reproducible except 7), e.g. G01 B cells (+ myeloid dendritic
  cells), G02 naive/CD4/CD8 T cells (T zone), G03 T/NK cells, G04 monocytes and dendritic cells,
  G06 plasma cells with stromal and mast cells (medullary cords), G07 endothelial and stromal cells.
- **Dominant type: 3,875 of 3,928 spots are `Mixed`** at the default `MIN_DOMINANT_FRACTION=0.4`
  (53 are B cell). With 20 fine-grained types (six kinds of T cell) and ~7 cells per spot, one type
  rarely holds 40 % of a spot. For a usable `dominant_type` on such a reference lower the threshold
  (even at 0.2, 2,482 spots stay `Mixed` here) or use a reference with coarser labels. `colocation_group`
  labels every spot instead: G01 1,428, G02 619, G03 601, G05 359, G06 293, G04 286, G07 264, G08 78.
  `abundance.csv` and `proportions.csv` are the primary output either way.
- The erythrocyte share (7 %) comes from haemoglobin transcripts in vessels and blood-filled
  sinuses; the reference has no germinal-centre B cells or follicular dendritic cells as separate
  types, so germinal centres appear as B cell (+ myeloid dendritic cell) spots.

## Limitations

- **The reference decides what can be found.** A cell type missing from the reference cannot be
  mapped; its RNA is attributed to the most similar types present (the demo reference has no
  follicular dendritic cells or germinal-centre B cells as separate types, so germinal centres
  appear as B cells). A reference from another tissue, species or protocol (single-nucleus vs
  single-cell) biases the abundances.
- **Abundances are estimates in "cells", not counts of nuclei.** They depend on
  `N_CELLS_PER_LOCATION` and `DETECTION_ALPHA`; compare abundances between spots and samples of
  one run rather than reading the absolute numbers literally. q05 is deliberately conservative.
- **Too few epochs under-fit.** With fewer than the tutorial's 30000 epochs the abundances are
  smoother and their totals lower; check `training_history.png` (the loss should have flattened)
  and `reconstruction_qc.png`.
- **Spot-based data only.** For single-cell platforms (Xenium, MERSCOPE, CosMx) use `st-celltype`.
  Training holds every spot × gene in memory at once (about 80 bytes per spot-gene entry);
  Visium HD bins at 8 µm (hundreds of thousands of bins) do not fit and are out of scope.
- `Mixed` is a threshold on the q05 shares, not a statement that a spot holds several types in
  equal amounts.
- Co-location groups are an NMF summary of the abundances, not spatial neighbourhoods; `st-niches`
  builds those from the abundances and the spot graph.

## Running it locally

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python      # pytest on the synthetic fixture (~1.5 min)
make run PYTHON=.venv/bin/python       # Docker over the fixture -> data/output/{labels,model}/
make validate PYTHON=.venv/bin/python  # app.yml against the manifest schema, CITATION.cff
docker build --platform linux/amd64 --build-arg TORCH_FLAVOUR=cpu  -t cell2location:cpu .
docker build --platform linux/amd64 --build-arg TORCH_FLAVOUR=cuda -t cell2location:cuda .
```

`sample-input/README.md` describes the fixture and how to run the lymph node demo. On a machine
with an NVIDIA GPU add `--gpus all` to `docker run` and `USE_GPU=auto` uses it.

## Licences

Our code is Apache-2.0 (`LICENSE`). cell2location is Apache-2.0, scvi-tools, scanpy and anndata
BSD-3-Clause, PyTorch BSD-style, Pyro and Lightning Apache-2.0. The image contains the NVIDIA CUDA
libraries of the CUDA build of PyTorch (proprietary, redistributable). Nothing in the image is
GPL. Details in `THIRD_PARTY.md`.

## Citations

Verified on Crossref and PubMed (title, authors, year); full author lists in `CITATION.cff`.

- **cell2location**: Kleshchevnikov V, Shmatko A, Dann E, Aivazidis A, King HW, Li T, Elmentaite R,
  Lomakin A, Kedlian V, Gayoso A, Jain MS, Park JS, Ramona L, Tuck E, Arutyunyan A, Vento-Tormo R,
  Gerstung M, James L, Stegle O, Bayraktar OA. Cell2location maps fine-grained cell types in
  spatial transcriptomics. *Nat Biotechnol* 2022;40(5):661-671. doi:10.1038/s41587-021-01139-4,
  PMID 35027729.
- **scvi-tools**: Gayoso A, Lopez R, Xing G, et al. A Python library for probabilistic analysis of
  single-cell omics data. *Nat Biotechnol* 2022;40(2):163-166. doi:10.1038/s41587-021-01206-w,
  PMID 35132262.
- **Scanpy**: Wolf FA, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data
  analysis. *Genome Biol* 2018;19:15. doi:10.1186/s13059-017-1382-0, PMID 29409532.
- **anndata**: Virshup I, Rybakov S, Theis FJ, Angerer P, Wolf FA. anndata: Access and store
  annotated data matrices. *J Open Source Softw* 2024;9(101):4371. doi:10.21105/joss.04371.
- **Demo reference, Tabula Sapiens**: The Tabula Sapiens Consortium. The Tabula Sapiens: A
  multiple-organ, single-cell transcriptomic atlas of humans. *Science* 2022;376(6594):eabl4896.
  doi:10.1126/science.abl4896, PMID 35549404.
- **Demo spatial data**: 10x Genomics, *Human Lymph Node* (Visium, Space Ranger 1.1.0), CC BY 4.0,
  https://www.10xgenomics.com/datasets/human-lymph-node-1-standard-1-1-0.
- Software without a paper DOI we could verify: Pyro (https://github.com/pyro-ppl/pyro), PyTorch
  (https://pytorch.org), scikit-learn (https://scikit-learn.org), Lightning
  (https://github.com/Lightning-AI/pytorch-lightning).
