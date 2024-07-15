use embedding_microservice::{run, Result};

#[tokio::main]
async fn main() -> Result<()> {
    run().await
}
