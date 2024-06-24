use std::collections::BTreeSet;

use std::fmt::Debug;
use std::fs::{self, File};
use std::marker::PhantomData;
use std::path::PathBuf;

use anyhow::Error as E;
use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::VarBuilderArgs;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::Repo;
use serde::{Deserialize, Deserializer};
use tokenizers::Tokenizer;
use tracing::info;

// https://huggingface.co/models?library=safetensors&other=bert&sort=trending

#[derive(Debug, Deserialize)]
struct Weightmaps {
    #[serde(deserialize_with = "deserialize_weight_map")]
    weight_map: BTreeSet<String>,
}
// Custom deserializer for the weight_map to directly extract values into a HashSet
fn deserialize_weight_map<'de, D>(deserializer: D) -> Result<BTreeSet<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let map = serde_json::Value::deserialize(deserializer)?;
    match map {
        serde_json::Value::Object(obj) => {
            let mut sorted_set = BTreeSet::new();
            for value in obj.values() {
                if let Some(v) = value.as_str() {
                    sorted_set.insert(v.to_string());
                }
            }
            Ok(sorted_set)
        }
        _ => Err(serde::de::Error::custom(
            "Expected an object for weight_map",
        )),
    }
}

pub struct ModelConfig {
    repo: ApiRepo,
    device: Device,
    fast: bool,
}

impl ModelConfig {
    fn hub_load_safetensors(&self) -> Result<Vec<PathBuf>> {
        let index_result = self.repo.get("model.safetensors.index.json");

        match index_result {
            Ok(json_file) => {
                let json_file = File::open(json_file)?;
                let json: Weightmaps =
                    serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;

                let pathbufs: Vec<PathBuf> = json
                    .weight_map
                    .iter()
                    .map(|f| self.repo.get(f).unwrap())
                    .collect();

                Ok(pathbufs)
            }
            Err(_) => {
                // If index file doesn't exist, try to load single safetensors file
                match self.repo.get("model.safetensors") {
                    Ok(file) => Ok(vec![file]),
                    Err(e) => Err(anyhow::anyhow!("Failed to load safetensors file: {}", e)),
                }
            }
        }
    }

    fn get_tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_filename = self.repo.get("tokenizer.json")?;

        Ok(Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?)
    }
    fn initialise_model(&self) -> Result<EmbeddingModel> {
        // Load the model configuration
        let config = fs::read_to_string(self.repo.get("config.json")?)?;
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = self.get_tokenizer()?;

        let filenames = self.hub_load_safetensors()?;

        if self.fast {
            config.hidden_act = HiddenAct::GeluApproximate;
        }

        let model = {
            let vb = unsafe {
                VarBuilderArgs::from_mmaped_safetensors(&filenames, DTYPE, &self.device)?
            };
            BertModel::load(vb, &config)?
        };
        Ok((model, tokenizer).into())
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        let model_id = "mixedbread-ai/mxbai-embed-large-v1";

        let repo = ApiBuilder::new()
            .build()
            .unwrap()
            .repo(Repo::model(model_id.to_string()));

        Self {
            repo,
            device: Device::Cpu,
            fast: false,
        }
    }
}

impl TryFrom<ModelConfig> for EmbeddingModel {
    fn try_from(value: ModelConfig) -> Result<Self> {
        value.initialise_model()
    }

    type Error = E;
}

impl ModelConfig {
    pub fn builder() -> ModelConfigBuilder<Unset, Unset> {
        ModelConfigBuilder::new()
    }
}
pub struct Unset;
pub struct Set;
pub struct ModelConfigBuilder<M, D> {
    model_id: Option<String>,
    device: Option<Device>,
    token: Option<String>,
    progress_bar: bool,
    cache_dir: Option<PathBuf>,
    fast: bool,
    _model_phantom: PhantomData<M>,
    _device_phantom: PhantomData<D>,
}

impl ModelConfigBuilder<Unset, Unset> {
    pub fn new() -> Self {
        Self {
            model_id: None,
            device: None,
            token: None,
            fast: false,
            progress_bar: false,
            cache_dir: None,
            _model_phantom: PhantomData,
            _device_phantom: PhantomData,
        }
    }
}

impl<D> ModelConfigBuilder<Unset, D> {
    pub fn model_id(self, model_id: impl Into<String>) -> ModelConfigBuilder<Set, D> {
        ModelConfigBuilder {
            model_id: Some(model_id.into()),
            device: self.device,
            token: self.token,
            fast: self.fast,
            progress_bar: self.progress_bar,
            cache_dir: self.cache_dir,
            _model_phantom: PhantomData,
            _device_phantom: self._device_phantom,
        }
    }
}

impl<M> ModelConfigBuilder<M, Unset> {
    pub fn device(self, cpu: bool) -> ModelConfigBuilder<M, Set> {
        let device = match Self::get_device(cpu) {
            Ok(device) => device,
            Err(e) => {
                println!("Failed to get device: {}. Defaulting to CPU.", e);
                Device::Cpu
            }
        };
        ModelConfigBuilder {
            model_id: self.model_id,
            device: Some(device),
            token: self.token,
            fast: self.fast,
            progress_bar: self.progress_bar,
            cache_dir: self.cache_dir,
            _model_phantom: self._model_phantom,
            _device_phantom: PhantomData,
        }
    }
}

impl ModelConfigBuilder<Set, Set> {
    pub fn token(mut self, token: impl Into<Option<String>>) -> Self {
        self.token = token.into();
        self
    }

    pub fn fast(mut self, fast: bool) -> Self {
        self.fast = fast;
        self
    }

    pub fn progress_bar(mut self, progress_bar: bool) -> Self {
        self.progress_bar = progress_bar;
        self
    }

    pub fn cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    pub fn build(self) -> ModelConfig {
        let mut builder = ApiBuilder::new()
            .with_token(self.token.into())
            .with_progress(self.progress_bar);

        if let Some(cache_dir) = self.cache_dir {
            builder = builder.with_cache_dir(cache_dir);
        }

        let api = builder.build().expect("Failed to build API");

        let model_id = self.model_id.expect("Model ID not set");

        ModelConfig {
            repo: api.repo(Repo::model(model_id)),
            device: self.device.expect("Device not set"),
            fast: self.fast,
        }
    }
}

impl<M, D> ModelConfigBuilder<M, D> {
    fn get_device(cpu: bool) -> Result<Device> {
        if cpu {
            Ok(Device::Cpu)
        } else if cuda_is_available() {
            Ok(Device::new_cuda(0)?)
        } else if metal_is_available() {
            Ok(Device::new_metal(0)?)
        } else {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                println!(
                    "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
                );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                println!(
                    "Running on CPU, to run on GPU, build this example with `--features cuda`"
                );
            }
            Ok(Device::Cpu)
        }
    }
}

pub struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
}

impl From<(BertModel, Tokenizer)> for EmbeddingModel {
    fn from((model, tokenizer): (BertModel, Tokenizer)) -> Self {
        Self { model, tokenizer }
    }
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        ModelConfig::default().try_into().unwrap()
    }
}

impl EmbeddingModel {
    pub fn device(&self) -> &Device {
        &self.model.device
    }

    pub fn new(model_config: ModelConfig) -> Self {
        let start = std::time::Instant::now();
        let model: EmbeddingModel = model_config.try_into().unwrap();
        println!("Loaded model in {:?}", start.elapsed());
        model
    }

    // pub fn encode<'s>(&self, input_text: impl Into<EncodeInput<'s>>) -> Result<Tensor> {
    //     let start = std::time::Instant::now();
    //     let tokens = self
    //         .tokenizer
    //         .encode(input_text, true)
    //         .map_err(E::msg)?
    //         .get_ids()
    //         .to_vec();

    //     let token_ids = Tensor::new(&tokens[..], self.device())?.unsqueeze(0)?;
    //     let token_type_ids = token_ids.zeros_like()?;
    //     println!("Encoded {:?}", start.elapsed());
    //     let start = std::time::Instant::now();
    //     let ys = self.model.forward(&token_ids, &token_type_ids)?;
    //     let ft = start.elapsed();
    //     println!("Forward {:?}", ft);
    //     println!("Dimensions: {:?}", ys.shape());
    //     println!("{ys}");
    //     // Mean pooling to get a single embedding vector
    //     let (_batch_size, seq_len, hidden_size) = ys.dims3()?;
    //     let pooled = { ys.sum(1)? / (seq_len as f64) }?;
    //     let pt = start.elapsed();
    //     println!("Pooled {:?}", pt - ft);

    //     println!("Pooled dimensions: {:?}", pooled.shape());
    //     println!("Total time {:?}", start.elapsed());
    //     Ok(pooled)
    // }

    fn tokenize(&self, sentences: &[&str]) -> Result<Tensor> {
        let mut tokenizer = self.tokenizer.clone();
        let tokenizer = tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        let tokens = tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)?;

        let input_ids = tokens
            .into_iter()
            .map(|enc| enc.get_ids().to_vec())
            .collect::<Vec<_>>();

        Ok(Tensor::new(input_ids, &self.device())?)
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let token_type_ids = input_ids.zeros_like()?;

        let embeddings = self.model.forward(input_ids, &token_type_ids)?;

        // Mean pooling to get a single embedding vector
        let seq_len = input_ids.dim(1)? as f64;
        { embeddings.sum(1)? / seq_len }.map_err(E::msg)
    }

    fn normalize_l2(&self, embeddings: &Tensor) -> Result<Tensor> {
        let norm = embeddings.sqr()?.sum_keepdim(1)?.sqrt()?;
        { embeddings / norm }.map_err(E::msg)
    }

    #[tracing::instrument(skip(self))]
    pub fn encode<I, S>(
        &self,
        sentences: I,
        batch_size: impl Into<Option<usize>> + Debug,
        normalize_embeddings: bool,
    ) -> Result<Tensor>
    where
        I: IntoIterator<Item = S> + Debug,
        S: AsRef<str>,
    {
        let start = std::time::Instant::now();
        let batch_size = batch_size.into().unwrap_or(32);
        let sentences = sentences.into_iter().collect::<Vec<_>>();

        // Sort sentences by length
        let mut sentence_lengths: Vec<_> = sentences
            .iter()
            .map(|s| s.as_ref().len())
            .enumerate()
            .collect();
        sentence_lengths.sort_unstable_by(|a, b| a.1.cmp(&b.1));

        let length_sorted_idx = sentence_lengths
            .iter()
            .map(|&(idx, _)| idx)
            .collect::<Vec<_>>();
        let sentences_sorted = length_sorted_idx
            .iter()
            .map(|&idx| sentences[idx].as_ref())
            .collect::<Vec<_>>();

        let mut all_embeddings = Vec::new();

        let batches = (sentences_sorted.len() + batch_size - 1) / batch_size;

        for batch in sentences_sorted.chunks(batch_size) {
            let start = std::time::Instant::now();
            let input_ids = self.tokenize(batch)?;
            let embeddings = self.forward(&input_ids)?;
            if batches > 1 {
                info!(
                    "Encoded batch of {:?} sentences in {:?}",
                    batch.len(),
                    start.elapsed()
                );
            }
            all_embeddings.push(embeddings);
        }

        let embeddings = Tensor::cat(&all_embeddings, 0)?;

        // Reorder embeddings to original order
        let mut reordered_embeddings =
            vec![
                Tensor::zeros(&[embeddings.dim(1)?], DType::F32, &self.device())?;
                embeddings.dim(0)?
            ];
        for (i, &idx) in length_sorted_idx.iter().enumerate() {
            reordered_embeddings[idx] = embeddings.get(i)?;
        }

        let embeddings = Tensor::stack(&reordered_embeddings, 0)?;

        println!(
            "Encoded {:?} sentences in {:?}",
            sentences.len(),
            start.elapsed()
        );

        if normalize_embeddings {
            self.normalize_l2(&embeddings)
        } else {
            Ok(embeddings)
        }
    }
}
