mod grpc_handlers;
mod http_handlers;
mod models;

use grpc_handlers::create_grpc_server;
use http_handlers::create_http_server;
use models::{EmbeddingModel, ModelConfig};

use std::{future::IntoFuture, sync::Arc};
use tokio::sync::Mutex;
use tracing::warn;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub type EmbeddingModelRef = Arc<Mutex<EmbeddingModel>>;

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

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    // HTTP service
    let http_server = create_http_server(listener, Arc::clone(&model)).into_future();

    // gRPC service
    let port = "0.0.0.0:50051".parse().unwrap();
    let grpc_server = create_grpc_server(port, Arc::clone(&model));

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
