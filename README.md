# Rust Sentence-Embeddings Microservice

This Rust project demonstrates a robust microservice architecture handling both HTTP and gRPC requests, designed to efficiently serve embedding models. Utilizing cutting-edge libraries and Rust's powerful features, this project showcases high performance and concurrency in network programming.

The microservice integrates machine learning models for sentence embeddings, allowing users to embed single or batch sentences via HTTP or gRPC. The service is built with modularity and scalability in mind, providing a clear example of how Rust can be used to create efficient and reliable network services.

## Features

- **HTTP and gRPC Servers**: Implemented using `axum` and `tonic` respectively, providing a clear example of Rust's capabilities in handling both HTTP and gRPC protocols within the same service.
- **Asynchronous Programming**: Utilizes `tokio`, showcasing effective use of asynchronous Rust for improved concurrency and scalability.
- **Machine Learning Model Integration**: Integrates machine learning models using `candle` libraries, illustrating how Rust can be used for high-performance ML tasks.
- **Configuration and Modular Design**: Structured with clear modularity and uses Cargo.toml for managing dependencies, highlighting best practices in Rust project organization and configuration.

## TODO

- [x] Load Bert model from Huggingface using `candle`

- [x] Handle models with a single `model.safetensor` file

- [x] Handle larger models with multiple `.safetensor` files

- [x] Handle single and batch embedding requests

- [x] Optimize batch performance

- [x] `HTTP` server using `axum`

- [x] Protocol Buffers for `gRPC` communication

- [x] `gRPC` server using `tonic`

- [x] Single and batch requests for both HTTP and gRPC

- [x] Working example clients for both HTTP and gRPC

- [ ] Add support for more architectures models, for example [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5/tree/main), [Salesforce/SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral)

- [ ] Better error handling

- [ ] Add more tests

## Getting Started

To get the microservice running locally:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/SamBroomy/rust-sentence-embeddings
   cd embedding-microservice
   ```

2. **Build and Run the server:**

   ```bash
   cargo build --release
   cargo run
   ```

This will run both the gRPC and HTTP servers. Pass cli arguments to specify the port and host for the server. If no port is passed both HTTP and gRPC servers will run on default ports `3000` and `50051` respectively.

If one is specified the server will run only the specified service on the specified port.

If both are specified the server will run both HTTP and gRPC servers on the specified ports.

```bash
cargo run -- --http-port 3000 --grpc-port 50051
```

### Supported Huggingface Models

Currently the service only supports [Safetensor BERT models from Huggingface](https://huggingface.co/models?library=safetensors&other=bert&sort=trending). I will look to add support for other embedding models in the future (like Mixtral)

The default model is `mixedbread-ai/mxbai-embed-large-v1` which is a high performing Bert model while being relatively small in size.

To change the model you can pass the model id as a cli argument.

```bash
cargo run -- --model-id "sentence-transformers/all-MiniLM-L6-v2"
```

Tested and working models are, but not limited to:

- `sentence-transformers/all-MiniLM-L6-v2` (All sentence-transformers models should work)
- `google-bert/bert-base-uncased` (All google bert models should work)
- `colbert-ir/colbertv2.0`
- `BAAI/bge-small-en-v1.5`

Basically any model that has BertModel as an architecture in the models `config.json` file should work. Looking to extend this to other architectures in the future.

```json
{
  ...
  "architectures": [
      "BertModel"
    ],
  ...
}
```

## Protocol Buffers and Client Examples

### Protocol Buffers

The `embedding_service.proto` file defines the service's protocol buffer schema for gRPC communication. This schema includes operations such as `Embed` and `EmbedBatch`, which are essential for the service's functionality.

To compile the protocol buffers into Rust code, a build script (`build.rs`) is used, which utilizes `tonic_build` to automate this process. Ensure that you have `tonic_build` and `prost` dependencies added in your `Cargo.toml` to successfully compile the `.proto` files.

Run the following command to compile the protocol buffers:

```bash
cargo build
```

### Running Client Examples

This project includes example client applications for both HTTP and gRPC interfaces, demonstrating how to interact with the microservice.

Be sure to have the server running before executing the client examples in a new terminal window.

#### HTTP Clients

- **Single Request:** To send a single embedding request using HTTP, run:

  ```bash
  cargo run --example http_client
  ```

- **Batch Request:** To send a batch of embedding requests using HTTP, run:

  ```bash
  cargo run --example http_client_batch
  ```

#### gRPC Clients

- **Single Request:** For sending a single embedding request via gRPC, use:

  ```bash
  cargo run --example grpc_client
  ```

- **Batch Request:** For sending a batch of embedding requests via gRPC, use:

  ```bash
  cargo run --example grpc_client_batch
  ```

Make sure the server is running before executing these example clients. The examples demonstrate how to use `reqwest` and `tonic` libraries effectively to interact with the server.

## Dependencies

- `axum` for HTTP server functionality
- `tonic` for gRPC server integration
- `candle-core`, `candle-nn`, and `candle-transformers` for machine learning capabilities
- `tokio` for asynchronous runtime
- Additional common Rust utilities like `serde`, `anyhow`, and `tracing` for serialization, error handling, and logging respectively.

## Improvements

If you spot any areas for improvement, feel free to let me know, always looking to improve and maintain best practices in Rust development.
