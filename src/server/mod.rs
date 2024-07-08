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

    let http_server = create_http_server(http_port, Arc::clone(&model)).into_future();
    let grpc_server = create_grpc_server(grpc_port, Arc::clone(&model));

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EmbeddingModel;
    use lazy_static::lazy_static;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    fn setup_model() -> Arc<Mutex<EmbeddingModel>> {
        Arc::new(Mutex::new(EmbeddingModel::default()))
    }

    fn setup_ports() -> (SocketAddr, SocketAddr) {
        let http_port: SocketAddr = "0.0.0.0:3000".parse().unwrap();
        let grpc_port: SocketAddr = "0.0.0.0:50051".parse().unwrap();
        (http_port, grpc_port)
    }
    lazy_static! {
        static ref MODEL: Arc<Mutex<EmbeddingModel>> = setup_model();
        static ref PORTS: (SocketAddr, SocketAddr) = setup_ports();
    }
    #[tokio::test]
    async fn test_run_http_server_a() {
        let port: SocketAddr = PORTS.0;
        let server = create_http_server(port, Arc::clone(&MODEL)).await;

        // Spawn the server in a background task
        let server_handle = tokio::spawn(async move {
            server.unwrap();
        });

        // Give the server some time to start
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        // Stop the server
        server_handle.abort();
    }
    #[tokio::test]
    async fn test_run_http_server() {
        let port: SocketAddr = PORTS.0;
        let model = Arc::clone(&MODEL);
        let server: axum::serve::Serve<axum::Router, axum::Router> =
            create_http_server(port, model);

        // Create a background task to run the server
        let server_handle = tokio::spawn(async move {
            server;
        });

        // Simulate some client requests or other interactions here if needed
        // For example, you could use a hyper client to send requests to your server

        // For this example, we'll just shut down the server after a short delay
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        // Shut down the server gracefully
        server_handle.abort();
        let _ = server_handle.await;
    }

    #[tokio::test]
    async fn test_run_grpc_server() {
        let port: SocketAddr = PORTS.1;
        let result = run_grpc_server(Arc::clone(&MODEL), port).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_both_servers() {
        let model = setup_model();
        let (http_port, grpc_port) = setup_ports();
        let result = run_both_servers(model, http_port, grpc_port).await;
        assert!(result.is_ok());

        drop(result);
    }
}
