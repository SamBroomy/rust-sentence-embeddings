use crate::Result;

use std::collections::BTreeSet;

use std::fmt::Debug;
use std::fs::{self, File};
use std::marker::PhantomData;
use std::path::PathBuf;

use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::VarBuilderArgs;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::Repo;
use serde::{de, Deserialize, Deserializer};
use tokenizers::Tokenizer;
use tracing::{error, info, instrument, span, warn};

// https://huggingface.co/models?library=safetensors&other=bert&sort=trending

#[derive(Debug, Deserialize)]
struct Weightmaps {
    #[serde(deserialize_with = "deserialize_weight_map")]
    weight_map: BTreeSet<String>,
}
// Custom deserializer for the weight_map to directly extract values into a HashSet
fn deserialize_weight_map<'de, D>(
    deserializer: D,
) -> std::result::Result<BTreeSet<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let map = serde_json::Value::deserialize(deserializer)?;
    match map {
        serde_json::Value::Array(arr) => {
            let mut sorted_set = BTreeSet::new();
            for value in arr {
                if let Some(v) = value.as_str() {
                    sorted_set.insert(v.to_string());
                }
            }
            Ok(sorted_set)
        }
        serde_json::Value::Object(obj) => {
            let mut sorted_set = BTreeSet::new();
            for value in obj.values() {
                if let Some(v) = value.as_str() {
                    sorted_set.insert(v.to_string());
                }
            }
            Ok(sorted_set)
        }
        _ => Err(de::Error::custom("Expected an object for weight_map")),
    }
}

pub struct ModelConfig {
    repo: ApiRepo,
    device: Device,
    fast: bool,
}

impl ModelConfig {
    #[instrument(skip_all, name="Loading Tensors" level = "debug")]
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
                let file = self.repo.get("model.safetensors")?;
                Ok(vec![file])
            }
        }
    }

    #[instrument(skip_all, name = "Loading Tokenizer", level = "debug")]
    fn get_tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_filename = self.repo.get("tokenizer.json")?;

        Tokenizer::from_file(tokenizer_filename).map_err(Into::into)
    }

    #[instrument(skip_all, name="Initialised Model" level = "warn")]

    fn initialise_model(&self) -> Result<EmbeddingModel> {
        // Load the model configuration
        let config = fs::read_to_string(self.repo.get("config.json")?)?;
        let mut config: Config = serde_json::from_str(&config)?;
        if self.fast {
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        let tokenizer = self.get_tokenizer()?;

        let filenames = self.hub_load_safetensors()?;

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
    type Error = crate::error::Error;
    fn try_from(value: ModelConfig) -> Result<Self> {
        value.initialise_model()
    }
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
    pub fn device(self, use_device: bool) -> ModelConfigBuilder<M, Set> {
        let device = match Self::get_device(use_device) {
            Ok(device) => device,
            Err(e) => {
                error!("Failed to get device: {}. Defaulting to CPU.", e);
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

    pub fn cache_dir(mut self, cache_dir: impl Into<Option<PathBuf>>) -> Self {
        self.cache_dir = cache_dir.into();
        self
    }

    pub fn build(self) -> ModelConfig {
        let mut builder = ApiBuilder::new()
            .with_token(self.token)
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
    fn get_device(use_device: bool) -> Result<Device> {
        if !use_device {
            info!("Running on CPU.");
            Ok(Device::Cpu)
        } else if cuda_is_available() {
            info!("CUDA is available, using GPU.");
            Ok(Device::new_cuda(0)?)
        } else if metal_is_available() {
            info!("Metal is available, using GPU.");
            Ok(Device::new_metal(0)?)
        } else {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                warn!(
                    "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
                );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                warn!("Running on CPU, to run on GPU, build this example with `--features cuda`");
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
        let model: EmbeddingModel = model_config.try_into().unwrap();
        model
    }

    #[instrument(skip_all, level = "debug")]
    fn tokenize(&self, sentences: &[&str]) -> Result<Tensor> {
        let mut tokenizer = self.tokenizer.clone();
        let tokenizer = tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        let tokens = tokenizer.encode_batch(sentences.to_vec(), true)?;

        let input_ids = tokens
            .into_iter()
            .map(|enc| enc.get_ids().to_vec())
            .collect::<Vec<_>>();

        Tensor::new(input_ids, self.device()).map_err(Into::into)
    }

    #[instrument(skip_all, level = "debug")]
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let token_type_ids = input_ids.zeros_like()?;

        let embeddings = self.model.forward(input_ids, &token_type_ids)?;

        // Mean pooling to get a single embedding vector
        let seq_len = input_ids.dim(1)? as f64;
        { embeddings.sum(1)? / seq_len }.map_err(Into::into)
    }

    #[instrument(skip(self), ret(Debug), name = "Encode Sentence")]
    pub fn encode<S>(&self, sentence: S) -> Result<Tensor>
    where
        S: AsRef<str> + Debug,
    {
        let input_ids = self.tokenize(&[sentence.as_ref()])?;
        Ok(self.forward(&input_ids)?.flatten_to(1)?)
    }

    #[instrument(
        skip(self, sentences),
        ret(Debug),
        fields(number_of_sentences),
        name = "Batch Encode"
    )]
    pub fn batch_encode<I, S>(
        &self,
        sentences: I,
        batch_size: impl Into<Option<usize>> + Debug,
    ) -> Result<Tensor>
    where
        I: IntoIterator<Item = S> + Debug,
        S: AsRef<str>,
    {
        let batch_size = match batch_size.into() {
            Some(size) => {
                if size > 0 {
                    size
                } else {
                    32
                }
            }
            None => 32,
        };
        assert!(batch_size > 0, "Batch size must be greater than 0");

        let sentences = sentences.into_iter().collect::<Vec<_>>();
        tracing::Span::current().record("number_of_sentences", sentences.len());

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

        for (idx, batch) in sentences_sorted.chunks(batch_size).enumerate() {
            let batch_span = span!(
                tracing::Level::DEBUG,
                "Encoding Batch",
                idx = idx,
                batches = batches,
                size = batch.len()
            );
            let _enter = batch_span.enter();
            let input_ids = self.tokenize(batch)?;
            let embeddings = self.forward(&input_ids)?;
            all_embeddings.push(embeddings);
        }

        let embeddings = Tensor::cat(&all_embeddings, 0)?;

        // Reorder embeddings to original order
        let mut reordered_embeddings =
            vec![
                Tensor::zeros(&[embeddings.dim(1)?], DType::F32, self.device())?;
                embeddings.dim(0)?
            ];
        for (i, &idx) in length_sorted_idx.iter().enumerate() {
            reordered_embeddings[idx] = embeddings.get(i)?;
        }

        Tensor::stack(&reordered_embeddings, 0).map_err(Into::into)
    }

    #[instrument(skip_all)]
    pub fn format_batch_embeddings(embeddings: Tensor) -> Result<Vec<Vec<f32>>> {
        embeddings.to_vec2::<f32>().map_err(Into::into)
    }

    #[instrument(skip_all)]
    pub fn format_embeddings(embeddings: Tensor) -> Result<Vec<f32>> {
        embeddings.to_vec1().map_err(Into::into)
    }
}
