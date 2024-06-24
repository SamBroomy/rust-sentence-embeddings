mod models;

use candle_transformers::models::whisper::model;
use models::ModelConfig;

fn main() {
    let model_name = "mixedbread-ai/mxbai-embed-large-v1";
    let config = ModelConfig::builder()
        .model_id(model_name)
        .device(false)
        .progress_bar(true)
        .build();
    let model = models::EmbeddingModel::new(config);

    let text = vec!["This is a test, i am testing the embedding model. I hope it works. Hello hello hello, ahhhhhhhh"];

    println!("Encoding text: {:?}", text);

    let embeddings = model.encode(&text, None, false).unwrap();

    println!("{}", embeddings);
}
