# syntax=docker/dockerfile:1.7

FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH} \
    LIBTORCH_USE_PYTORCH=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    curl \
    git \
    mold \
    protobuf-compiler \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY scripts/ scripts/

RUN chmod +x scripts/rustc-wrapper.sh

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    CARGO_BUILD_RUSTC_WRAPPER=/app/scripts/rustc-wrapper.sh \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=clang \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS='-C link-arg=-fuse-ld=/usr/bin/mold' \
    cargo build --release -p hydra-train && \
    cp /app/target/release/train /tmp/train && \
    cp /app/target/release/mjai_audit /tmp/mjai_audit && \
    cp /app/target/release/recompress /tmp/recompress

FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    LIBTORCH_USE_PYTORCH=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN python - <<'PY'
import os
import torch

libdir = os.path.join(os.path.dirname(torch.__file__), 'lib')
with open('/etc/ld.so.conf.d/pytorch-libtorch.conf', 'w', encoding='utf-8') as fh:
    fh.write(libdir + '\n')
PY

RUN ldconfig

WORKDIR /app

COPY --from=builder /tmp/train /usr/local/bin/train
COPY --from=builder /tmp/mjai_audit /usr/local/bin/mjai_audit
COPY --from=builder /tmp/recompress /usr/local/bin/recompress

ENTRYPOINT ["/usr/local/bin/train"]
