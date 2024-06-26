use crate::models::EmbeddingModel;
use crate::EmbeddingModelRef;

use candle_core::Tensor;
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::transport::{Error, Server};
use tonic::{IntoRequest, Request, Response, Status};
use tracing::{debug, instrument, span};
use tracing_subscriber::field::debug;

pub mod embedding_service {
    tonic::include_proto!("embedding");

    impl From<Vec<f32>> for Embedding {
        fn from(values: Vec<f32>) -> Self {
            Embedding { values }
        }
    }

    impl From<Embedding> for Vec<f32> {
        fn from(embedding: Embedding) -> Self {
            embedding.values
        }
    }

    impl From<Vec<Vec<f32>>> for EmbedBatchResponse {
        fn from(embeddings: Vec<Vec<f32>>) -> Self {
            EmbedBatchResponse {
                embeddings: embeddings.into_iter().map(Embedding::from).collect(),
            }
        }
    }

    impl From<EmbedBatchResponse> for Vec<Vec<f32>> {
        fn from(response: EmbedBatchResponse) -> Self {
            response.embeddings.into_iter().map(Vec::from).collect()
        }
    }
}

pub struct EmbeddingServiceImpl {
    model: EmbeddingModelRef,
}

impl EmbeddingServiceImpl {
    pub fn new(model: EmbeddingModelRef) -> Self {
        Self { model }
    }
}

use embedding_service::embedding_service_server::{EmbeddingService, EmbeddingServiceServer};
use embedding_service::{EmbedBatchRequest, EmbedBatchResponse, EmbedRequest, Embedding};

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServiceImpl {
    #[instrument(skip_all, name = "gRPC Embed")]
    async fn embed(&self, request: Request<EmbedRequest>) -> Result<Response<Embedding>, Status> {
        let request = request.into_inner();
        let text = request.text;
        let embedding: Tensor;
        {
            let model = self.model.lock().await;
            embedding = model
                .encode(text)
                .map_err(|e| Status::internal(format!("Error encoding text: {:?}", e)))?;
        }

        let cpu_span = span!(tracing::Level::DEBUG, "Embedding to CPU");
        let _cpu_guard = cpu_span.enter();

        let cpu_embedding = EmbeddingModel::to_cpu_async(embedding)
            .await
            .map_err(|e| Status::internal(format!("Error moving embeddings to CPU: {:?}", e)))?;

        let embedding_output = EmbeddingModel::format_embeddings_async(Arc::new(cpu_embedding))
            .await
            .map_err(|e| Status::internal(format!("Error formatting embeddings: {:?}", e)))?;
        drop(_cpu_guard);

        Ok(Response::new(Embedding::from(embedding_output)))
    }

    #[instrument(skip_all, name = "gRPC Embed Batch")]
    async fn embed_batch(
        &self,
        request: Request<EmbedBatchRequest>,
    ) -> Result<Response<EmbedBatchResponse>, Status> {
        let request = request.into_inner();
        let texts = request.texts;
        let batch_size = request.batch_size;
        let batch_size = if request.batch_size == 0 {
            None
        } else {
            Some(batch_size as usize)
        };

        let embeddings: Tensor;

        {
            let model = self.model.lock().await;
            embeddings = model
                .batch_encode(texts, batch_size)
                .map_err(|e| Status::internal(format!("Error encoding batch text: {:?}", e)))?;
        }

        let cpu_span = span!(tracing::Level::DEBUG, "Embedding to CPU");
        let _cpu_guard = cpu_span.enter();

        debug!("embeddings: {:?}", embeddings);

        let chunks = embeddings.dim(0).unwrap() as usize;
        debug!("chunks: {}", chunks);

        let embeddings = embeddings.chunk(chunks, 0).unwrap();

        debug!("embeddings: {:?}", embeddings);

        let embeddings = EmbeddingModel::to_cpu_async_batch(embeddings)
            .await
            .map_err(|e| Status::internal(format!("Error moving embeddings to CPU: {:?}", e)))?;

        let embeddings_output = embeddings
            .into_iter()
            .map(|t| t.to_vec1::<f32>().unwrap())
            .collect::<Vec<_>>();

        // let embeddings_output = EmbeddingModel::format_batch_embeddings_async(Arc::new(embeddings))
        //     .map_err(|e| Status::internal(format!("Error formatting batch embeddings: {:?}", e)))?;
        // drop(_cpu_guard);

        Ok(Response::new(EmbedBatchResponse::from(embeddings_output)))
    }
}

fn build_service(model: EmbeddingModelRef) -> EmbeddingServiceImpl {
    // gRPC service
    EmbeddingServiceImpl::new(Arc::clone(&model))
}

pub fn create_grpc_server(
    port: SocketAddr,
    model: EmbeddingModelRef,
) -> impl std::future::Future<Output = Result<(), Error>> {
    let grpc_service = build_service(model);

    Server::builder()
        .add_service(EmbeddingServiceServer::new(grpc_service))
        .serve(port)
}
