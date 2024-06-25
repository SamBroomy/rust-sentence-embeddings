/// This example demonstrates how to use the `reqwest` crate to send a POST request to the server. Ensure that the server is running before running this client.
/// ```sh
/// cargo run```
/// New terminal window
/// ```sh
/// cargo run --example http_client```
use reqwest::Client;
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let texts = "This is a test sentence.".to_string();

    let response = client
        // Can be get or post
        .post("http://localhost:3000/embed")
        .json(&json!({ "text": texts}))
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
