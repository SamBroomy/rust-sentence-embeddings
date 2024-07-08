pub type Result<T> = std::result::Result<T, Error>;
//pub type Error = Box<dyn std::error::Error>;

// Wtf is going on here, remove this and it thinks its dead code but its not is it?
#[allow(dead_code)]
// I dont understand why the Error types within the Error enum are giving me this dead code warning? I think it could be something to do with Deserialize on the struct Weightmaps or the deserialize_weight_map function. But i literally have no clue. Any help with this would be appreciated.
#[derive(Debug, derive_more::From)]
pub enum Error {
    SerdeDe(String),
    #[from]
    Candle(candle_core::Error),
    #[from]
    HfApi(hf_hub::api::sync::ApiError),
    #[from]
    Io(std::io::Error),
    #[from]
    Tokenizer(tokenizers::Error),
    #[from]
    Tonic(tonic::transport::Error),
    #[from]
    Serde(serde_json::Error),
    #[from]
    Anyhow(anyhow::Error),
}

// Error boilerplate.
impl core::fmt::Display for Error {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(fmt, "{self:?}")
    }
}

impl std::error::Error for Error {}

//For custom error messages.
// impl Error {
//     pub fn serde_de_error(val: impl std::fmt::Display) -> Self {
//         Self::SerdeDeError(val.to_string())
//     }
// }

// impl From<&str> for Error {
//     fn from(val: &str) -> Self {
//         Self::SerdeDeError(val.into())
//     }
// }
