# Faster builds (after the first time) https://depot.dev/docs/languages/rust-dockerfile

FROM rust:1.79 AS base
RUN cargo install --locked cargo-chef sccache
ENV RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache


FROM base AS planner
WORKDIR /app
# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml Cargo.lock ./

# Copy the rest of the source code
COPY src ./src
COPY proto ./proto
COPY build.rs ./

RUN --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef prepare --recipe-path recipe.json


FROM base as builder
WORKDIR /app

# Install system dependencies
RUN apt update && apt upgrade -y && apt-get install --reinstall build-essential
RUN apt install -y \
    protobuf-compiler \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo build --release


# Start a new stage for a smaller final image
# FROM rust:1.79-slim
# # 671MB

FROM debian:bookworm-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt update && apt upgrade -y && \
    apt install -y \
    ca-certificates \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /app/target/release/embedding-microservice /usr/local/bin/


# Set the startup command
CMD ["embedding-microservice"]