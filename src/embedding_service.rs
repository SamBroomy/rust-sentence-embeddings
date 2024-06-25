use crate::models::EmbeddingModel;
use candle_core::Tensor;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

pub mod embedding_service {
    tonic::include_proto!("embedding");
}

use embedding_service::embedding_service_server::EmbeddingService;
use embedding_service::{EmbedBatchRequest, EmbedBatchResponse, EmbedRequest, Embedding};

impl From<Vec<f32>> for Embedding {
    fn from(values: Vec<f32>) -> Self {
        Self { values }
    }
}

pub struct EmbeddingServiceImpl {
    model: Arc<Mutex<EmbeddingModel>>,
}

impl EmbeddingServiceImpl {
    pub fn new(model: Arc<Mutex<EmbeddingModel>>) -> Self {
        Self { model }
    }
}

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServiceImpl {
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
        let embedding_vec = EmbeddingModel::format_embeddings(embedding)
            .map_err(|e| Status::internal(format!("Error formatting embeddings: {:?}", e)))?;

        Ok(Response::new(embedding_vec.into()))
    }

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
        let embeddings_vec = EmbeddingModel::format_batch_embeddings(embeddings)
            .map_err(|e| Status::internal(format!("Error formatting batch embeddings: {:?}", e)))?;

        Ok(Response::new(EmbedBatchResponse {
            embeddings: embeddings_vec.into_iter().map(Embedding::from).collect(),
        }))
    }
}
