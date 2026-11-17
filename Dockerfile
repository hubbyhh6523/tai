# Not built/tested in the authoring sandbox (no Docker there).
FROM rust:1-slim-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/embed-svc /usr/local/bin/embed-svc
ENV PORT=8090 MODEL_DIR=/model MODEL_NAME=bge-small-zh-v1.5 DIM=512 THREADS=4
EXPOSE 8090
HEALTHCHECK CMD ["bash","-c","exec 3<>/dev/tcp/127.0.0.1/${PORT} && printf 'GET /health HTTP/1.0\\r\\n\\r\\n' >&3 && grep -q 200 <&3"]
CMD ["embed-svc"]
