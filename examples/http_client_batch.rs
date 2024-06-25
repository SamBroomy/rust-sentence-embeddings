/// This example demonstrates how to use the `reqwest` crate to send a batch POST request to the server. Ensure that the server is running before running this client.
/// ```sh
/// cargo run```
/// New terminal window
/// ```sh
/// cargo run --example http_client_batch```
use reqwest::Client;
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let texts = vec![
        "This is a test sentence.".to_string(),
        "Another example for embedding.".to_string(),
    ];

    let response = client
        .post("http://localhost:3000/batch_embed")
        .json(&json!({ "texts": texts, "batch_size": 0 }))
        .send()
        .await?;

    if response.status().is_success() {
        let result: Value = response.json().await?;
        println!("Embeddings: {}", result);
    } else {
        println!("Error: {:?}", response.status());
    }

    Ok(())
}
