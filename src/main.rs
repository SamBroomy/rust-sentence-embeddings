mod embedding_service;
mod models;

use candle_core::Tensor;
use embedding_service::{
    embedding_service::embedding_service_server::EmbeddingServiceServer, EmbeddingServiceImpl,
};
use models::{EmbeddingModel, ModelConfig};

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{future::IntoFuture, sync::Arc};
use tokio::sync::Mutex;
use tonic::transport::Server;
use tracing::{info, warn};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Serialize, Deserialize)]
struct TextRequest {
    text: String,
}

#[derive(Serialize, Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct TextBatchRequest {
    texts: Vec<String>,
    batch_size: usize,
}

#[derive(Serialize, Deserialize)]
struct EmbeddingBatchResponse {
    embeddings: Vec<Vec<f32>>,
}

type EmbeddingService = Arc<Mutex<EmbeddingModel>>;

async fn handle_embeddings(
    State(model): State<EmbeddingService>,
    Json(payload): Json<TextRequest>,
) -> std::result::Result<Json<EmbeddingResponse>, StatusCode> {
    info!("Received HTTP Embedding request: [{:?}]", payload.text);

    let embeddings: Tensor;

    {
        let model = model.lock().await;
        embeddings = model.encode(&payload.text).map_err(|e| {
            tracing::error!("Error encoding text: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    }

    let embeddings_vec = embeddings.to_vec1().map_err(|e| {
        tracing::error!("Error converting embeddings: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(EmbeddingResponse {
        embedding: embeddings_vec,
    }))
}

async fn handle_batch_embeddings(
    State(model): State<EmbeddingService>,
    Json(payload): Json<TextBatchRequest>,
) -> std::result::Result<Json<EmbeddingBatchResponse>, StatusCode> {
    info!(
        "Received Batch Embedding HTTP request: {:?}",
        payload.texts.len()
    );

    let embeddings: Tensor;

    {
        let model = model.lock().await;
        embeddings = model.batch_encode(&payload.texts, None).map_err(|e| {
            tracing::error!("Error encoding text: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    }

    let embeddings_vec = embeddings.to_vec2().map_err(|e| {
        tracing::error!("Error converting embeddings: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(EmbeddingBatchResponse {
        embeddings: embeddings_vec,
    }))
}

async fn hello() -> &'static str {
    info!("Received request!");
    "Hello, World!"
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let model_name = "mixedbread-ai/mxbai-embed-large-v1";
    let config = ModelConfig::builder()
        .model_id(model_name)
        .device(false)
        .progress_bar(true)
        .build();
    let model = models::EmbeddingModel::new(config);
    let model = Arc::new(Mutex::new(model));

    let http_app = Router::new()
        .route("/", get(hello))
        .route("/embed", get(handle_embeddings).post(handle_embeddings))
        .route("/batch_embed", post(handle_batch_embeddings))
        .with_state(Arc::clone(&model));

    // gRPC service
    let grpc_service = EmbeddingServiceImpl::new(Arc::clone(&model));
    let grpc_server = Server::builder()
        .add_service(EmbeddingServiceServer::new(grpc_service))
        .serve("0.0.0.0:50051".parse().unwrap());

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    let http_server = axum::serve(listener, http_app).into_future();

    let text = ["This is a test, i am testing the embedding model. I hope it works. Hello hello hello, ahhhhhhhh", "Another test, would you look at that!"];
    {
        let embeddings = model.lock().await.batch_encode(&text, None)?;

        let _ = embeddings.to_vec2::<f32>()?;
    }
    {
        let embeddings = model.lock().await.encode("This is a test, i am testing the embedding model. I hope it works. Hello hello hello, ahhhhhhhh")?;
        let _ = embeddings.to_vec1::<f32>()?;
    }

    warn!("Starting servers...");

    // Run both servers concurrently
    tokio::select! {
        _ = grpc_server => warn!("gRPC server terminated"),
        _ = http_server => warn!("HTTP server terminated"),
    }

    Ok(())
}
