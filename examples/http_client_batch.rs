/// This example demonstrates how to use the `reqwest` crate to send a batch POST request to the server. Ensure that the server is running before running this client.
/// ```sh
/// cargo run```
/// New terminal window
/// ```sh
/// cargo run --example http_client_batch```
mod utils;

use reqwest::Client;
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let texts = utils::read_in_batch_examples();

    let start = std::time::Instant::now();

    let response = client
        .post("http://localhost:3000/batch_embed")
        .json(&json!({ "texts": texts, "batch_size": 0 }))
        .send()
        .await?;

    if response.status().is_success() {
        let result: Value = response.json().await?;

        // Extract the `embeddings` key
        if let Some(embeddings) = result.get("embeddings") {
            if let Ok(embeddings) = serde_json::from_value::<Vec<Vec<f32>>>(embeddings.clone()) {
                println!(
                    "Time taken for request and conversion to Vec<Vec<f32>>: {:?}",
                    start.elapsed()
                );

                println!("Received {} embeddings", embeddings.len());
                println!("Inner dimension: {}", embeddings.first().unwrap().len());
            }
        }
    } else {
        println!("Error: {:?}", response.status());
    }

    Ok(())
}
