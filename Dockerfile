# Not built/tested in the authoring sandbox (no Docker there).
FROM rust:1-slim-trixie AS build
# Optional crates.io mirror (sparse index at <mirror>/index/), e.g. https://rsproxy.cn
ARG CARGO_REGISTRY_MIRROR=
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++-14-dev && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN if [ -n "$CARGO_REGISTRY_MIRROR" ]; then \
      printf '[source.crates-io]\nreplace-with = "mirror"\n[source.mirror]\nregistry = "sparse+%s/index/"\n[net]\nretry = 3\n' \
        "$CARGO_REGISTRY_MIRROR" > "${CARGO_HOME:-/usr/local/cargo}/config.toml"; \
    fi \
 && cargo build --release

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/embed-svc /usr/local/bin/embed-svc
COPY deploy/fetch-model.sh /usr/local/bin/fetch-model.sh
RUN chmod +x /usr/local/bin/fetch-model.sh \
    && /usr/local/bin/fetch-model.sh /model \
    && rm /usr/local/bin/fetch-model.sh
ENV PORT=8090 MODEL_DIR=/model MODEL_NAME=bge-small-zh-v1.5 DIM=512 THREADS=4
EXPOSE 8090
HEALTHCHECK CMD ["bash","-c","exec 3<>/dev/tcp/127.0.0.1/${PORT} && printf 'GET /health HTTP/1.0\\r\\n\\r\\n' >&3 && grep -q 200 <&3"]
CMD ["embed-svc"]
