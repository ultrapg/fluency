use std::io::Write;
use std::path::{Path, PathBuf};

use crate::settings::ModelSize;

fn model_filename(size: ModelSize) -> &'static str {
    match size {
        ModelSize::Tiny => "ggml-tiny.bin",
        ModelSize::TinyEn => "ggml-tiny.en.bin",
        ModelSize::Base => "ggml-base.bin",
        ModelSize::BaseEn => "ggml-base.en.bin",
        ModelSize::Small => "ggml-small.bin",
        ModelSize::SmallEn => "ggml-small.en.bin",
        ModelSize::Medium => "ggml-medium.bin",
        ModelSize::MediumEn => "ggml-medium.en.bin",
        ModelSize::LargeV3 => "ggml-large-v3.bin",
        ModelSize::LargeV3Turbo => "ggml-large-v3-turbo.bin",
    }
}

fn model_url(size: ModelSize) -> String {
    let base = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
    format!("{base}/{}", model_filename(size))
}

pub fn path_for_size(size: ModelSize) -> PathBuf {
    let data_dir = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
    data_dir.join("fluency").join(model_filename(size))
}

pub fn ensure_downloaded_by_size(size: ModelSize) -> anyhow::Result<PathBuf> {
    let path = path_for_size(size);
    if path.exists() {
        return Ok(path);
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let url = model_url(size);
    eprint!("Downloading {} model ({}MB)... ", size.name(), size.size_mb());
    std::io::stderr().flush()?;

    let response = ureq::get(&url)
        .call()
        .map_err(|e| anyhow::anyhow!("Failed to download {} model: {e}", size.name()))?;

    let mut file = std::fs::File::create(&path)?;
    let mut reader = response.into_reader();
    let n = std::io::copy(&mut reader, &mut file)
        .map_err(|e| anyhow::anyhow!("Failed to write model file: {e}"))?;

    let mb = n as f64 / 1_048_576.0;
    eprintln!("done ({mb:.1} MB)");

    Ok(path)
}

pub fn default_tiny_path() -> PathBuf {
    path_for_size(ModelSize::Tiny)
}

pub fn ensure_downloaded(path: &Path) -> anyhow::Result<()> {
    if path.exists() {
        return Ok(());
    }

    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let size = if name.contains("tiny.en") { ModelSize::TinyEn }
        else if name.contains("tiny") { ModelSize::Tiny }
        else if name.contains("base.en") { ModelSize::BaseEn }
        else if name.contains("base") { ModelSize::Base }
        else if name.contains("small.en") { ModelSize::SmallEn }
        else if name.contains("small") { ModelSize::Small }
        else if name.contains("medium.en") { ModelSize::MediumEn }
        else if name.contains("medium") { ModelSize::Medium }
        else if name.contains("large-v3-turbo") { ModelSize::LargeV3Turbo }
        else if name.contains("large") { ModelSize::LargeV3 }
        else { ModelSize::Tiny };

    ensure_downloaded_by_size(size)?;
    Ok(())
}
