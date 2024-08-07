export RUST_BACKTRACE := "1"

# Variables

release := "false"
release_flag := if release == "true" { "--release" } else { "" }
image_name := "embedding-microservice"
http_port := "3000"
grpc_port := "50051"

alias b := build
alias r := run
alias bcr := build_container
alias rcr := run_container

# List all the available commands
[private]
default:
    @just --list --unsorted

# Build the rust project
[group('rust')]
[macos]
build:
    cargo build {{ release_flag }}

# Run the rust project
[group('rust')]
[macos]
run:
    cargo run {{ release_flag }} --features metal

# Run the rust project
[group('rust')]
[linux]
[windows]
run:
    cargo run {{ release_flag }} --features cuda

# Run the rust project with the grpc example
[group('rust')]
run_example_grpc:
    cargo run --example grpc_client

# Run the rust project with the grpc_batch example
[group('rust')]
run_example_grpc_batch:
    cargo run -example grpc_client_batch

# Run the rust project with the http example
[group('rust')]
run_example_http:
    cargo run --example http_client

# Run the rust project with the http_batch example
[group('rust')]
run_example_http_batch:
    cargo run --example http_client_batch

# Run all the examples (ensure the server is running)
[group('rust')]
run_examples: run_example_grpc run_example_grpc_batch run_example_http run_example_http_batch

# Build the docker container
[group('docker')]
build_container:
    docker build -t {{ image_name }} .

# Run the docker container with HTTP and gRPC ports
[group('docker')]
run_container http_port=http_port grpc_port=grpc_port: stop_container build_container
    docker run -p {{ http_port }}:{{ http_port }} -p {{ grpc_port }}:{{ grpc_port }} {{ image_name }}

# Run the docker container with HTTP port
[group('docker')]
run_grpc_container port=grpc_port: build_container
    docker run -p {{ port }}:{{ port }} {{ image_name }} --grpc-port {{ port }}

# Run the docker container with gRPC port
[group('docker')]
run_http_container port=http_port: build_container
    docker run -p {{ port }}:{{ port }} {{ image_name }} --http-port {{ port }}

# Stop all running containers based on the microservice image
[group('docker')]
stop_container:
    # List all running containers based on the embedding-microservice image and stop them
    docker ps -q --filter "ancestor=embedding-microservice" | xargs -r docker stop

# Clean the project
[group('docker')]
[group('rust')]
clean: stop_container
    cargo clean
    docker rm $(docker ps -a -q --filter "ancestor={{ image_name }}")

# Purge all images based on the microservice image
[confirm]
[group('docker')]
purge_all: clean
    docker rmi $(docker images -q --filter "ancestor={{ image_name }}")
