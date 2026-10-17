ARG RUST_IMAGE
ARG CUDA_DEVEL_IMAGE
ARG CUDA_RUNTIME_IMAGE
FROM ${RUST_IMAGE} AS rust
FROM ${CUDA_DEVEL_IMAGE} AS build
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:/opt/quasar/venv/bin:$PATH
ENV CUDA_PATH=/usr/local/cuda AUGUR_CUDA_GENCODE=arch=compute_70,code=sm_70
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3 python3-venv ca-certificates libssl-dev && rm -rf /var/lib/apt/lists/*
COPY . /src
RUN python3 -m venv /opt/augur/venv && python -m pip install --no-index --no-deps --require-hashes --find-links=/src/runtime-wheels -r /src/requirements.lock
RUN python -m pip check
WORKDIR /src
RUN python -B scripts/build_augur_windows.py --stage build --feature augur-model-qwen35-runtime/qwen35-augur-vision --out /opt/augur/build --target-dir /tmp/augur-target
RUN python -B scripts/build_augur_windows.py --stage cpu --out /opt/augur/build
RUN cargo fetch --locked --target x86_64-unknown-linux-gnu
RUN python -B scripts/prepare_augur_notices.py --context /src --cargo-cache /usr/local/cargo/registry/cache --out /opt/augur/notices
FROM ${CUDA_RUNTIME_IMAGE} AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends python3 libssl3t64 ca-certificates && rm -rf /var/lib/apt/lists/* && useradd --uid 10001 --create-home augur && mkdir /evidence && chown augur:augur /evidence
COPY --from=build /opt/augur /opt/augur
COPY container/augur/entry.py /opt/augur/entry.py
ENV PATH=/opt/augur/venv/bin:$PATH CUDA_PATH=/usr/local/cuda
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV AUGUR_BF16_STARTUP=streamed AUGUR_HTTP_DECODE_ATTENTION=tile8_b1
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER 10001:10001
WORKDIR /opt/augur/build/source
STOPSIGNAL SIGTERM
ENTRYPOINT ["/opt/augur/venv/bin/python", "-B", "/opt/augur/entry.py"]
