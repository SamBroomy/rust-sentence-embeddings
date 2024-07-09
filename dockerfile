# Use the official Rust image as a parent image
FROM rust:1.79 as builder

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt update && apt upgrade -y && apt-get install --reinstall build-essential
RUN apt install -y \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*


# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml Cargo.lock ./

# Copy the rest of the source code
COPY src ./src
COPY proto ./proto
COPY build.rs ./
# Build the application
RUN cargo build --release --keep-going


# Start a new stage for a smaller final image
FROM rust:1.79-slim

# Install runtime dependencies
# RUN apt update && apt upgrade -y && \
#     #apt install -y \
#     #ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /usr/src/app/target/release/embedding-microservice /usr/local/bin/


# Set the startup command
CMD ["embedding-microservice"]