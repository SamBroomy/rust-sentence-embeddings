/// This is an example of a gRPC client that sends a batch request to the embedding service. Ensure that the service is running before running this client.
/// ```sh
/// cargo run```
/// In new terminal window
/// ```sh
/// cargo run --example grpc_client_batch```
mod utils;
use utils::embedding_service::{
    embedding_service_client::EmbeddingServiceClient, EmbedBatchRequest,
};

use tonic::Request;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbeddingServiceClient::connect("http://localhost:50051").await?;

    let texts = utils::read_in_batch_examples();

    let batch_size = 0;

    let start = std::time::Instant::now();

    let request = Request::new(EmbedBatchRequest {
        texts: texts,
        batch_size,
    });

    match client.embed_batch(request).await {
        Ok(response) => {
            let embeddings: Vec<Vec<f32>> = response.into_inner().into();

            println!(
                "Time taken for request and conversion to Vec<Vec<f32>>: {:?}",
                start.elapsed()
            );

            println!("Received {} embeddings", embeddings.len());
            println!("Inner dimension: {}", embeddings.first().unwrap().len());
        }

        Err(e) => println!("Error: {:?}", e),
    }

    Ok(())
}
