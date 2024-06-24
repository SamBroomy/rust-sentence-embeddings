use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct EmbeddingRequest {
    texts: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct EmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

async fn get_embeddings(texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
    let client = Client::new();
    let request = EmbeddingRequest { texts };

    let response = client
        .post("http://localhost:3000/embed")
        .body(serde_json::to_string(&request)?)
        .send()
        .await?;

    let embeddings = serde_json::from_str::<EmbeddingResponse>(&response.text().await?)?;

    Ok(embeddings.embeddings)
}

#[tokio::main]
async fn main() -> Result<()> {
    let texts = vec!["Hello, world!".to_string(), "This is a test.".to_string()];

    match get_embeddings(texts).await {
        Ok(embeddings) => {
            println!("Received embeddings:");
            for (i, embedding) in embeddings.iter().enumerate() {
                println!("Embedding {}: {:?}", i, embedding);
            }
        }
        Err(e) => eprintln!("Error getting embeddings: {}", e),
    }

    Ok(())
}
