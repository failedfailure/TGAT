FROM python:3.11-slim-bookworm

ARG UNIMIXLM_REV=ec6d1db61f5c1ed374717b1c88af2db54235655d
ARG RUST_VERSION=1.92.0
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false HF_HOME=/workspace/.hf-cache \
    CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup \
    PATH=/opt/venv/bin:/opt/cargo/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates cmake curl git pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv \
    && curl --fail --silent --show-error --location https://sh.rustup.rs \
       | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_VERSION}"

RUN git init /opt/unimixlm \
    && git -C /opt/unimixlm remote add origin https://github.com/cimeister/unimixlm.git \
    && git -C /opt/unimixlm fetch --depth 1 origin "${UNIMIXLM_REV}" \
    && git -C /opt/unimixlm checkout --detach FETCH_HEAD \
    && git -C /opt/unimixlm submodule update --init --recursive \
    && test "$(git -C /opt/unimixlm/tokenizers rev-parse HEAD)" = 3afc67f12e19725ac7167e4be999d8c28ad8f22a \
    && test "$(git -C /opt/unimixlm/sentencepiece rev-parse HEAD)" = 2b7ec9b8e86a61f7772236471f948850872d8918

COPY requirements.txt /tmp/tgat-requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 \
    && python -m pip install --no-cache-dir maturin==1.11.5 \
    && python -m pip install --no-cache-dir --no-build-isolation \
       /opt/unimixlm/tokenizers/bindings/python -r /tmp/tgat-requirements.txt \
    && python -m pip check \
    && cmake -S /opt/unimixlm/sentencepiece -B /tmp/spm-build \
       -DCMAKE_INSTALL_PREFIX=/usr/local -DSPM_ENABLE_SHARED=OFF \
       -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && cmake --build /tmp/spm-build --parallel 2 \
    && cmake --install /tmp/spm-build \
    && rm -rf /tmp/spm-build \
    && (spm_train --help 2>&1 || true) | grep -q seed_sentencepieces_file

WORKDIR /workspace
COPY tgat_phase1/ /workspace/tgat_phase1/
