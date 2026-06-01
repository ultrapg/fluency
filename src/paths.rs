use std::path::PathBuf;

pub fn whisper_model_path(model_file: &str) -> PathBuf {
    let data_dir = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
    data_dir.join("fluency").join(model_file)
}
