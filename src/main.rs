mod error;
mod models;
mod server;

use error::Result;
use models::{EmbeddingModel, ModelConfig};
use server::run_service;
use server::EmbeddingModelRef;

use clap::Parser;
use std::path::PathBuf;
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::Mutex;
use tracing::info;
use tracing::Level;
use tracing_subscriber::fmt::format::FmtSpan;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        default_value = "mixedbread-ai/mxbai-embed-large-v1",
        help = "The full huggingface model id"
    )]
    model_id: String,

    #[arg(
        short,
        long,
        default_value_t = true,
        help = "Use a device for inference (e.g. GPU) or not (CPU). Default to cpu if device not available."
    )]
    device: bool,

    #[arg(long, help = "HTTP port to listen on")]
    http_port: Option<u16>,

    #[arg(long, help = "gRPC port to listen on")]
    grpc_port: Option<u16>,

    #[arg(
        short,
        long,
        default_value_t = true,
        help = "Show progress bar for downloads"
    )]
    progress_bar: bool,

    #[arg(
        short,
        long,
        default_value_t = false,
        help = "Use GeluApproximate activation function for faster inference."
    )]
    fast: bool,

    #[arg(short, long)]
    token: Option<String>,

    #[arg(short, long)]
    cache_dir: Option<PathBuf>,

    #[arg(short, long, default_value = "INFO")]
    level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let level = match args.level.to_uppercase().as_str() {
        "DEBUG" => Level::DEBUG,
        "INFO" => Level::INFO,
        "WARN" => Level::WARN,
        "ERROR" => Level::ERROR,
        _ => {
            eprintln!("Invalid log level: {}", args.level);
            std::process::exit(1);
        }
    };

    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_max_level(level)
        .init();

    let config = ModelConfig::builder()
        .model_id(&args.model_id)
        .device(args.device)
        .progress_bar(args.progress_bar)
        .token(args.token)
        .fast(args.fast)
        .cache_dir(args.cache_dir)
        .build();

    let http_port: Option<SocketAddr> = args
        .http_port
        .map(|port| format!("0.0.0.0:{}", port).parse().unwrap());

    let grpc_port: Option<SocketAddr> = args
        .http_port
        .map(|port| format!("0.0.0.0:{}", port).parse().unwrap());

    let model = models::EmbeddingModel::new(config);
    let model = Arc::new(Mutex::new(model));

    test_model(&model).await?;

    run_service(http_port, grpc_port, model).await?;

    Ok(())
}

async fn test_model(model: &EmbeddingModelRef) -> Result<()> {
    let text = [
        "This is a test, i am testing the embedding model. I hope it works. Hello hello hello, ahhhhhhhh",
        "Another test, would you look at that!",
    ];

    {
        let embeddings = model.lock().await.encode(text[0])?;
        let _ = embeddings.to_vec1::<f32>()?;
    }
    {
        let embeddings = model.lock().await.batch_encode(text, None)?;
        let _ = embeddings.to_vec2::<f32>()?;
    }

    info!("Embedding model test successful");
    Ok(())
}
