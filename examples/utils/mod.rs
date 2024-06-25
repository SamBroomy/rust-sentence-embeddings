pub fn read_in_batch_examples() -> Vec<String> {
    let file = std::fs::read_to_string("./examples/data/batch_examples.txt").unwrap();
    let examples: Vec<String> = file.lines().map(|s| s.to_string()).collect();
    examples
}

pub mod embedding_service {
    tonic::include_proto!("embedding");

    // duplication of the Embedding struct, just continence methods but any way to de duplicate? or is that kinda the point?
    // Because its just protobuf generated code, and we can't modify it and will need to add convenience methods in our own code.

    impl From<Vec<f32>> for Embedding {
        fn from(values: Vec<f32>) -> Self {
            Embedding { values }
        }
    }

    impl From<Embedding> for Vec<f32> {
        fn from(embedding: Embedding) -> Self {
            embedding.values
        }
    }

    impl From<Vec<Vec<f32>>> for EmbedBatchResponse {
        fn from(embeddings: Vec<Vec<f32>>) -> Self {
            EmbedBatchResponse {
                embeddings: embeddings.into_iter().map(Embedding::from).collect(),
            }
        }
    }

    impl From<EmbedBatchResponse> for Vec<Vec<f32>> {
        fn from(response: EmbedBatchResponse) -> Self {
            response.embeddings.into_iter().map(Vec::from).collect()
        }
    }
}
