# Third-party software

This app is Apache-2.0 (see `LICENSE`). The image it builds contains the following third-party
components, installed from PyPI at the pinned versions in `requirements.txt`; their licences
apply to the image. Nothing here is GPL (no igraph or leidenalg: this app does not cluster). The
open-source parts are under permissive licences (Apache-2.0, BSD, MIT, PSF, MPL-2.0; MPL-2.0 is
file-level copyleft and leaves our code untouched). Two images are built (`TORCH_FLAVOUR`): the
**cpu** variant has the CPU-only PyTorch wheel (`2.14.1+cpu`) and no NVIDIA software; in the **gpu**
variant, the **NVIDIA CUDA libraries** that come with the CUDA build of PyTorch are proprietary but redistributable under the NVIDIA software licence
terms shipped in each wheel (the same arrangement as every CUDA PyTorch image, including
processor-lst-ai's base image).

| Component | What | Licence | Source |
|---|---|---|---|
| cell2location 0.1.5 | `RegressionModel` (reference signatures), `Cell2location` (spatial mapping), `CoLocatedGroupsSklearnNMF` (co-location groups), `utils.filtering.filter_genes`. Called as a library, not modified. | Apache-2.0 | https://github.com/BayraktarLab/cell2location |
| scvi-tools 1.5.2 | Model/training framework under cell2location (`setup_anndata`, training loop, posterior sampling). | BSD-3-Clause | https://github.com/scverse/scvi-tools |
| PyTorch 2.14.1 (cpu variant: `+cpu` build; gpu variant: `+cu126` build with triton 3.8.0) | Tensor computation on CPU or GPU. | BSD-3-Clause style (torch's licence bundle: Apache-2.0, BSD-2/3-Clause, BSL-1.0, ISC, MIT, Zlib); triton MIT | https://pytorch.org |
| nvidia-*-cu12 (cublas, cuda-cupti, cuda-nvrtc, cuda-runtime, cudnn, cufft, cufile, curand, cusolver, cusparse, cusparselt, nccl, nvjitlink, nvshmem, nvtx; versions in `requirements-cuda.txt`); cuda-toolkit 12.6.3, cuda-bindings 12.9.10, cuda-pathfinder 1.8.5 | gpu variant only: CUDA runtime libraries (unused on CPU). | NVIDIA proprietary, redistributable (licence in each wheel); cuda-bindings/pathfinder Apache-2.0 | https://developer.nvidia.com/cuda-toolkit |
| Pyro 1.9.1 (pyro-ppl, pyro-api 0.1.2) | Probabilistic programming (variational inference) the cell2location models are written in. | Apache-2.0 | https://github.com/pyro-ppl/pyro |
| Lightning 2.6.6 (lightning, pytorch-lightning, lightning-utilities 0.15.3, torchmetrics 1.9.0) | Training loop used by scvi-tools. | Apache-2.0 | https://github.com/Lightning-AI/pytorch-lightning |
| scanpy 1.12.5 | Imported by cell2location and scvi-tools. | BSD-3-Clause | https://github.com/scverse/scanpy |
| anndata 0.13.4, mudata 0.4.1 | Reads the `.h5ad` files; data containers of scvi-tools. | BSD-3-Clause | https://github.com/scverse |
| scikit-learn 1.9.1 | `NMF` inside cell2location's co-location model. | BSD-3-Clause | https://scikit-learn.org |
| NumPy 2.5.4, SciPy 1.18.1 | Arrays, sparse matrices, `linear_sum_assignment` (aligning NMF restarts). | BSD-3-Clause (NumPy also bundles 0BSD/MIT/Zlib/CC0 parts) | https://numpy.org, https://scipy.org |
| pandas 3.1.0, h5py 3.16.0 | Tables; HDF5 backend of anndata. | BSD-3-Clause | https://pandas.pydata.org, https://www.h5py.org |
| Matplotlib 3.11.2, Pillow 13.0.0 | PNG figures; image arrays. | Matplotlib licence (PSF-based, BSD-compatible); MIT-CMU | https://matplotlib.org, https://python-pillow.org |
| xarray 2026.9.0, sparse 0.19.3, ml_collections 1.1.0, docrep 0.3.2, rich 15.0.0, tensorboard 2.21.0 (+ tensorboard-data-server, absl-py, grpcio, protobuf, Markdown, Werkzeug), pydantic 2.14.2 (+ pydantic-core, pydantic-settings, annotated-types, typing-inspection, python-dotenv), formulaic 1.2.2, interface_meta, narwhals, msgspec, opt_einsum, cloudpickle, fsspec, aiohttp (+ aiosignal, aiohappyeyeballs, frozenlist, multidict, propcache, yarl, attrs, idna), Jinja2, MarkupSafe, sympy, mpmath, networkx, filelock, setuptools, certifi | Dependencies of scvi-tools, torch and Lightning (versions in `requirements.txt`). | Apache-2.0 / BSD / MIT / MPL-2.0 (certifi) | PyPI |
| numba 0.68.1, llvmlite 0.50.0, umap-learn 0.5.12, pynndescent 0.7.0, statsmodels 0.15.0, patsy 1.0.3, seaborn 0.13.2, scikit-misc 0.5.3, joblib, threadpoolctl, natsort, fast-array-utils, legacy-api-wrap, session-info2, scverse-misc, zarr, numcodecs, donfig, google-crc32c | Dependencies of scanpy / anndata. | BSD-2/3-Clause, MIT, Apache-2.0, MPL-2.0 (fast-array-utils, legacy-api-wrap, session-info2) | PyPI |
| contourpy, cycler, fonttools, kiwisolver, packaging, pyparsing, python-dateutil, six, tqdm, typing_extensions, tzdata, wrapt, PyYAML, Pygments, markdown-it-py, mdurl | Dependencies of the above. | MIT / BSD / Apache-2.0 / MPL-2.0+MIT (tqdm) / PSF-2.0 | PyPI |
| python:3.12.16-slim-bookworm | Base image (Debian 12 + CPython 3.12). | PSF (Python); Debian packages under their own free licences | https://hub.docker.com/_/python |

`opencv-python`, a declared dependency of cell2location, is **not** installed: it is only used by
`cell2location.plt.RotateCrop` (histology rotation in notebooks), which this app never imports.

## Not bundled

- **Demo data** (not in the image or the repository): 10x Genomics *Human Lymph Node* (Visium,
  Space Ranger 1.1.0; CC BY 4.0) and the Tabula Sapiens lymph node scRNA-seq data (The Tabula
  Sapiens Consortium, Science 2022, doi:10.1126/science.abl4896; CELLxGENE collection
  e5f58829-1a66-40b5-a624-9046778e74f5, CC BY 4.0), subsampled to at most 500 cells per type.
