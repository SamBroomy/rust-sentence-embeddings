mod grpc_handlers;
mod http_handlers;

use crate::EmbeddingModel;
use crate::Result;

use grpc_handlers::create_grpc_server;
use http_handlers::create_http_server;
use std::{future::IntoFuture, net::SocketAddr, sync::Arc};
use tracing::info;

use tokio::sync::Mutex;
use tracing::warn;

pub type EmbeddingModelRef = Arc<Mutex<EmbeddingModel>>;

async fn run_http_server(model: EmbeddingModelRef, port: SocketAddr) -> Result<()> {
    warn!("Starting HTTP server...");
    let server = create_http_server(port, model);
    info!("HTTP server listening on: {:}", port);
    Ok(server.await?)
}

async fn run_grpc_server(model: EmbeddingModelRef, port: SocketAddr) -> Result<()> {
    warn!("Starting gRPC server...");
    let server = create_grpc_server(port, model);
    info!("gRPC server listening on: {:}", port);
    server.await.map_err(Into::into)
}

async fn run_both_servers(
    model: EmbeddingModelRef,
    http_port: SocketAddr,
    grpc_port: SocketAddr,
) -> Result<()> {
    warn!("Starting both HTTP and gRPC servers...");
    info!("HTTP server listening on: {:}", http_port);
    info!("gRPC server listening on: {:}", grpc_port);

    let http_server = create_http_server(http_port, model.clone()).into_future();
    let grpc_server = create_grpc_server(grpc_port, model.clone());

    warn!("Starting both servers...");

    tokio::select! {
        _ = grpc_server => warn!("gRPC server terminated"),
        _ = http_server => warn!("HTTP server terminated"),
    }

    Ok(())
}

pub async fn run_service(
    http_port: Option<SocketAddr>,
    grpc_port: Option<SocketAddr>,
    model: EmbeddingModelRef,
) -> Result<()> {
    match (http_port, grpc_port) {
        (Some(http_port), Some(grpc_port)) => {
            run_both_servers(model, http_port, grpc_port).await?;
        }
        (Some(http_port), None) => {
            run_http_server(model, http_port).await?;
        }
        (None, Some(grpc_port)) => {
            run_grpc_server(model, grpc_port).await?;
        }
        (None, None) => {
            let http_port: SocketAddr = "0.0.0.0:3000".parse().unwrap();
            let grpc_port: SocketAddr = "0.0.0.0:50051".parse().unwrap();

            run_both_servers(model, http_port, grpc_port).await?;
        }
    }

    Ok(())
}
