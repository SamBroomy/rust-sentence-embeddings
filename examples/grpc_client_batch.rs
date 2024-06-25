/// This is an example of a gRPC client that sends a request to the embedding service. Ensure that the service is running before running this client.
/// ```sh
/// cargo run --example grpc_client```
use tonic::Request;

pub mod embedding_service {
    tonic::include_proto!("embedding");
}

use embedding_service::embedding_service_client::EmbeddingServiceClient;
use embedding_service::EmbedBatchRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbeddingServiceClient::connect("http://localhost:50051").await?;

    let texts = [
        "This is a test sentence.".to_string(),
        "This is another test sentence.".to_string(),
    ];

    let batch_size = 0;

    let request = Request::new(EmbedBatchRequest {
        texts: texts.into(),
        batch_size,
    });

    match client.embed_batch(request).await {
        Ok(response) => {
            let embeddings = response.into_inner().embeddings;
            println!("Received {} embeddings", embeddings.len());
            for (i, embedding) in embeddings.iter().enumerate() {
                println!("Received embedding {}: {:?}", i, embedding);
            }
        }

        Err(e) => println!("Error: {:?}", e),
    }

    Ok(())
}
