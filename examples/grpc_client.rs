/// This is an example of a gRPC client that sends a request to the embedding service. Ensure that the service is running before running this client.
/// ```sh
/// cargo run --example grpc_client```
use tonic::Request;

pub mod embedding_service {
    tonic::include_proto!("embedding");
}

use embedding_service::embedding_service_client::EmbeddingServiceClient;
use embedding_service::EmbedRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbeddingServiceClient::connect("http://localhost:50051").await?;

    let text = "This is a test sentence.".to_string();

    let request = Request::new(EmbedRequest { text });

    match client.embed(request).await {
        Ok(response) => {
            let response = response.into_inner();

            println!("Received embedding: {:?}", response.values);
        }
        Err(e) => println!("Error: {:?}", e),
    }

    Ok(())
}
