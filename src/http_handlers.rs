use candle_core::Tensor;

use crate::EmbeddingModelRef;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::info;

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
    batch_size: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct EmbeddingBatchResponse {
    embeddings: Vec<Vec<f32>>,
}

async fn handle_embeddings(
    State(model): State<EmbeddingModelRef>,
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
    State(model): State<EmbeddingModelRef>,
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

fn build_router(model: EmbeddingModelRef) -> Router {
    Router::new()
        .route("/", get(hello))
        .route("/embed", get(handle_embeddings).post(handle_embeddings))
        .route("/batch_embed", post(handle_batch_embeddings))
        .with_state(Arc::clone(&model))
}

pub fn create_http_server(
    tcp_listener: TcpListener,
    model: EmbeddingModelRef,
) -> axum::serve::Serve<Router, Router> {
    let http_app = build_router(model);
    axum::serve(tcp_listener, http_app)
}
