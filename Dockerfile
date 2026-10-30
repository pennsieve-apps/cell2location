FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# The complete pinned dependency set (no resolver: see requirements.txt; wheels, and the pure-Python
# sdists of cell2location and a few small packages). torch is the
# CUDA 13.2 build from PyPI, like processor-lst-ai: the same image runs on the standard (cpu) runtime
# and uses the GPU on the gpu runtime. It makes the image large (about 7 GB).
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-deps --prefer-binary -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -path '*/torch/*' -prune -o -type d -name tests -prune -exec rm -rf {} +

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/cache

# Build matplotlib's font cache and import the stack once as the run-time user would, so a run needs no
# writes outside /tmp and no network.
RUN python -c "import matplotlib.pyplot, scanpy, cell2location, cell2location.models.downstream" \
    && mkdir -p /tmp/matplotlib /tmp/numba /tmp/cache && chmod -R a+rwX /tmp/matplotlib /tmp/numba /tmp/cache

# Runs as uid 1000 on the platform; nothing here needs root.
USER 1000

ENTRYPOINT ["python3", "-m", "processor.entry"]
