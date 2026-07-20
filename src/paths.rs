use std::path::PathBuf;

pub fn whisper_model_path(model_file: &str) -> PathBuf {
    if let Ok(mut exe_path) = std::env::current_exe() {
        exe_path.pop();
        exe_path.join("models").join(model_file)
    } else {
        PathBuf::from("models").join(model_file)
    }
}

pub fn llm_model_path(model_file: &str) -> PathBuf {
    if let Ok(mut exe_path) = std::env::current_exe() {
        exe_path.pop();
        exe_path.join("models").join(model_file)
    } else {
        PathBuf::from("models").join(model_file)
    }
}
