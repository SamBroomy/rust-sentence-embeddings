mod models;

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Extension, Json, Router,
};
use models::{EmbeddingModel, ModelConfig};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::info;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Serialize, Deserialize)]
struct TextRequest {
    texts: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct EmbeddingsResponse {
    embeddings: Vec<Vec<f32>>,
}

// async fn handle_embeddings(
//     Json(payload): Json<TextRequest>,
//     Extension(model): Extension<Arc<Mutex<EmbeddingModel>>>,
// ) -> impl IntoResponse {
//     match handle_embeddings_inner(payload, model).await {
//         Ok(response) => (StatusCode::OK, response),
//         Err(status) => (status, Json(EmbeddingsResponse { embeddings: vec![] })),
//     }
//     .into_response()
// }

// async fn handle_embeddings_inner(
//     payload: TextRequest,
//     model: Arc<Mutex<EmbeddingModel>>,
// ) -> std::result::Result<Json<EmbeddingsResponse>, StatusCode> {
//     let model = model.lock().await; // Lock the model for exclusive access
//     let embeddings = model.encode(&payload.texts, None, false).map_err(|e| {
//         tracing::error!("Error encoding text: {:?}", e);
//         StatusCode::INTERNAL_SERVER_ERROR
//     })?;
//     Ok(Json(EmbeddingsResponse {
//         embeddings: embeddings.to_vec2::<f32>().map_err(|e| {
//             tracing::error!("Error converting embeddings: {:?}", e);
//             StatusCode::INTERNAL_SERVER_ERROR
//         })?,
//     }))
// }

async fn handle_embeddings(
    State(model): State<Arc<Mutex<EmbeddingModel>>>,
    Json(payload): Json<TextRequest>,
) -> std::result::Result<Json<EmbeddingsResponse>, StatusCode> {
    info!("Received request: {:?}", payload.texts.len());

    let model = model.lock().await;
    let embeddings = model.encode(&payload.texts, None, false).map_err(|e| {
        tracing::error!("Error encoding text: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let embeddings_vec = embeddings.to_vec2().map_err(|e| {
        tracing::error!("Error converting embeddings: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(EmbeddingsResponse {
        embeddings: embeddings_vec,
    }))
}

async fn hello() -> &'static str {
    info!("Received request!");
    "Hello, World!"
}

#[tokio::main]
async fn main() -> Result<()> {
    // initialize tracing
    tracing_subscriber::fmt::init();

    let model_name = "mixedbread-ai/mxbai-embed-large-v1";
    let config = ModelConfig::builder()
        .model_id(model_name)
        .device(false)
        .progress_bar(true)
        .build();
    let model = models::EmbeddingModel::new(config);
    let model = Arc::new(Mutex::new(model));

    let app = Router::new()
        .route("/", get(hello))
        .route("/embed", post(handle_embeddings))
        .with_state(model);

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();

    //let text = ["This is a test, i am testing the embedding model. I hope it works. Hello hello hello, ahhhhhhhh", "Another test, would you look at that!"];

    //println!("Encoding text: {:?}", text);

    //let embeddings = model.lock().await.encode(&text, None, false)?;
    //println!("Embeddings: {}", embeddings);
    //let embeddings = embeddings.to_vec2::<f32>()?;

    Ok(())
}
