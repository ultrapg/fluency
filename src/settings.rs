use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSize {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    LargeV3,
    LargeV3Turbo,
}

impl ModelSize {
    pub fn all() -> &'static [ModelSize] {
        &[
            ModelSize::Tiny,
            ModelSize::TinyEn,
            ModelSize::Base,
            ModelSize::BaseEn,
            ModelSize::Small,
            ModelSize::SmallEn,
            ModelSize::Medium,
            ModelSize::MediumEn,
            ModelSize::LargeV3Turbo,
            ModelSize::LargeV3,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            ModelSize::Tiny => "tiny",
            ModelSize::TinyEn => "tiny.en",
            ModelSize::Base => "base",
            ModelSize::BaseEn => "base.en",
            ModelSize::Small => "small",
            ModelSize::SmallEn => "small.en",
            ModelSize::Medium => "medium",
            ModelSize::MediumEn => "medium.en",
            ModelSize::LargeV3 => "large-v3",
            ModelSize::LargeV3Turbo => "large-v3-turbo",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            ModelSize::Tiny => "Fastest, 75 MB, lowest accuracy",
            ModelSize::TinyEn => "Fastest, English-only, 75 MB",
            ModelSize::Base => "Fast, 150 MB, decent accuracy",
            ModelSize::BaseEn => "Fast, English-only, 150 MB",
            ModelSize::Small => "Balanced, 470 MB, good accuracy",
            ModelSize::SmallEn => "Balanced, English-only, 470 MB",
            ModelSize::Medium => "Slower, 1.5 GB, very good accuracy",
            ModelSize::MediumEn => "Slower, English-only, 1.5 GB",
            ModelSize::LargeV3 => "Best accuracy, 3 GB, slowest",
            ModelSize::LargeV3Turbo => "Fast + accurate, 1.5 GB, distilled v3",
        }
    }

    pub fn size_mb(&self) -> &'static str {
        match self {
            ModelSize::Tiny | ModelSize::TinyEn => "75 MB",
            ModelSize::Base | ModelSize::BaseEn => "150 MB",
            ModelSize::Small | ModelSize::SmallEn => "470 MB",
            ModelSize::Medium | ModelSize::MediumEn => "1.5 GB",
            ModelSize::LargeV3 => "3 GB",
            ModelSize::LargeV3Turbo => "1.5 GB",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSettings {
    pub normalize: bool,
    pub highpass_enabled: bool,
    pub highpass_cutoff: f32,
    pub noise_gate_enabled: bool,
    pub noise_gate_threshold: f32,
    pub preemphasis_enabled: bool,
    pub preemphasis_coefficient: f32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            normalize: true,
            highpass_enabled: true,
            highpass_cutoff: 80.0,
            noise_gate_enabled: true,
            noise_gate_threshold: 0.005,
            preemphasis_enabled: false,
            preemphasis_coefficient: 0.97,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptStyle {
    #[serde(rename = "clean")]
    Clean,
    #[serde(rename = "bullets")]
    BulletPoints,
    #[serde(rename = "smart")]
    Smart,
    #[serde(rename = "minimal")]
    Minimal,
}

impl Default for PromptStyle {
    fn default() -> Self { PromptStyle::Smart }
}

impl PromptStyle {
    pub fn all() -> &'static [PromptStyle] {
        &[PromptStyle::Clean, PromptStyle::BulletPoints, PromptStyle::Smart, PromptStyle::Minimal]
    }

    pub fn name(&self) -> &'static str {
        match self {
            PromptStyle::Clean => "Clean paragraphs",
            PromptStyle::BulletPoints => "Bullet points",
            PromptStyle::Smart => "Smart formatting",
            PromptStyle::Minimal => "Minimal (caps + punctuation)",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            PromptStyle::Clean => "Full paragraphs with filler removal. Adds capital letters, periods, paragraph breaks, and removes hesitations.",
            PromptStyle::BulletPoints => "Converts the text into a bullet-point list with one item per line (each starting with '- '). Good for notes.",
            PromptStyle::Smart => "Analyzes the text and only fixes what needs improvement: punctuation, capitalization, filler words, and topic breaks. Keeps the original wording.",
            PromptStyle::Minimal => "Only adds ending punctuation and fixes capitalization of the first letter. Does not change or remove any words.",
        }
    }

    pub fn tooltip(&self) -> &'static str {
        match self {
            PromptStyle::Clean => "The LLM reformats your transcription into clean paragraphs with proper punctuation, capitalization, line breaks between topics, and removes filler words like 'um' and 'uh'. Best for formal writing or sharing.",
            PromptStyle::BulletPoints => "The LLM rewrites the text as a bullet-point list so each idea is on its own line. Great for brainstorming notes or to-do lists.",
            PromptStyle::Smart => "The LLM checks each sentence and only adds what's missing: capital letters, periods, commas, and paragraph breaks. Filler words are removed. Sentences are NOT rewritten — every word stays. This is the default and works best for general use.",
            PromptStyle::Minimal => "The LLM only adds a period at the end of sentences and capitalizes the first letter. No words are added, removed, or changed. Best if you want pure raw transcription with basic readability.",
        }
    }
}

fn default_fillers() -> Vec<String> {
    vec![
        "um".into(),
        "uh".into(),
        "like".into(),
        "you know".into(),
        "hmm".into(),
        "er".into(),
        "ah".into(),
    ]
}

fn default_correction_markers() -> Vec<String> {
    vec!["or".into(), "i mean".into(), "no".into(), "actually".into()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatSettings {
    #[serde(default = "default_true")]
    pub bigram: bool,
    #[serde(default = "default_true")]
    pub auto_capitalize: bool,
    #[serde(default = "default_true")]
    pub auto_punctuate: bool,
    #[serde(default)]
    pub remove_fillers: bool,
    #[serde(default)]
    pub fix_corrections: bool,
    #[serde(default = "default_true")]
    pub lm_correction: bool,
    #[serde(default = "default_fillers")]
    pub fillers: Vec<String>,
    #[serde(default = "default_correction_markers")]
    pub correction_markers: Vec<String>,
}

const fn default_true() -> bool { true }

impl Default for FormatSettings {
    fn default() -> Self {
        Self {
            bigram: true,
            auto_capitalize: true,
            auto_punctuate: true,
            remove_fillers: false,
            fix_corrections: false,
            lm_correction: true,
            fillers: default_fillers(),
            correction_markers: default_correction_markers(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingSettings {
    // Sampling strategy: "greedy" or "beam_search"
    pub sampling_strategy: String,
    pub beam_size: i32,
    pub patience: f32,
    pub best_of: i32,

    // Decoder tuning
    pub temperature: f32,
    pub temperature_inc: f32,
    pub suppress_blank: bool,
    pub suppress_nst: bool,
    pub no_speech_thold: f32,
    pub length_penalty: f32,
    pub entropy_thold: f32,
    pub logprob_thold: f32,

    // Context & prompt
    pub initial_prompt: String,
    pub no_context: bool,
    pub n_max_text_ctx: i32,

    // Experimental
    pub single_segment: bool,
    pub audio_ctx: i32,
    pub debug_mode: bool,
    pub max_tokens: i32,

    // Threads
    pub n_threads: i32,

    // Voice Activity Detection (built-in whisper.cpp VAD)
    pub vad_enabled: bool,
    pub vad_threshold: f32,
    pub vad_min_speech_duration_ms: i32,
    pub vad_min_silence_duration_ms: i32,
    pub vad_max_speech_duration_s: f32,
    pub vad_speech_pad_ms: i32,
    pub vad_samples_overlap: f32,
}

const DEFAULT_INITIAL_PROMPT: &str =
    "Hello, how are you today? I'm doing great, thanks for asking! Let me check on that. \
     The meeting is at 3:00 PM. I'll send you the report by Friday. What do you think? \
     Perfect, that sounds good. Actually, I need to reconsider. On second thought, let's go with the \
     original plan. I want 50, not 30. Can you help me with this? Yes, that would be wonderful. \
     I'm going to the store. Do you need anything? Let me know what you think.";

impl Default for ProcessingSettings {
    fn default() -> Self {
        Self {
            sampling_strategy: "beam_search".to_string(),
            beam_size: 5,
            patience: -1.0,
            best_of: 5,
            temperature: 0.0,
            temperature_inc: 0.2,
            suppress_blank: true,
            suppress_nst: true,
            no_speech_thold: 0.6,
            length_penalty: -1.0,
            entropy_thold: 2.4,
            logprob_thold: -1.0,
            initial_prompt: DEFAULT_INITIAL_PROMPT.to_string(),
            no_context: false,
            n_max_text_ctx: 16384,
            single_segment: false,
            audio_ctx: 0,
            debug_mode: false,
            max_tokens: 0,
            n_threads: 0,
            vad_enabled: false,
            vad_threshold: 0.5,
            vad_min_speech_duration_ms: 250,
            vad_min_silence_duration_ms: 100,
            vad_max_speech_duration_s: f32::MAX,
            vad_speech_pad_ms: 30,
            vad_samples_overlap: 0.1,
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmModelId {
    #[serde(rename = "gemma3_270m")]
    Gemma3_270M,
    #[serde(rename = "smollm2_360m")]
    SmolLM2_360M,
    #[serde(rename = "qwen2_5_0_5b")]
    Qwen2_5_0_5B,
    #[serde(rename = "tinyllama_1_1b")]
    TinyLlama_1_1B,
    #[serde(rename = "smollm2_1_7b")]
    SmolLM2_1_7B,
}

impl LlmModelId {
    pub fn all() -> &'static [LlmModelId] {
        &[
            // Gemma3_270M removed — crashes with buffer overflow on bundled llama.cpp
            LlmModelId::SmolLM2_360M,
            LlmModelId::Qwen2_5_0_5B,
            LlmModelId::TinyLlama_1_1B,
            LlmModelId::SmolLM2_1_7B,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            LlmModelId::Gemma3_270M => "Gemma 3 270M",
            LlmModelId::SmolLM2_360M => "SmolLM2 360M",
            LlmModelId::Qwen2_5_0_5B => "Qwen2.5 0.5B",
            LlmModelId::TinyLlama_1_1B => "TinyLlama 1.1B",
            LlmModelId::SmolLM2_1_7B => "SmolLM2 1.7B",
        }
    }

    pub fn hf_repo(&self) -> &'static str {
        match self {
            LlmModelId::Gemma3_270M => "lmstudio-community/gemma-3-270m-it-GGUF",
            LlmModelId::SmolLM2_360M => "bartowski/SmolLM2-360M-Instruct-GGUF",
            LlmModelId::Qwen2_5_0_5B => "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            LlmModelId::TinyLlama_1_1B => "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            LlmModelId::SmolLM2_1_7B => "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        }
    }

    pub fn gguf_filename(&self) -> &'static str {
        match self {
            LlmModelId::Gemma3_270M => "gemma-3-270m-it-Q4_K_M.gguf",
            // Q4_0 instead of Q4_K_M — K-quants cause buffer overflow on bundled llama.cpp (ARM)
            LlmModelId::SmolLM2_360M => "SmolLM2-360M-Instruct-Q4_0.gguf",
            LlmModelId::Qwen2_5_0_5B => "qwen2.5-0.5b-instruct-q4_0.gguf",
            LlmModelId::TinyLlama_1_1B => "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
            LlmModelId::SmolLM2_1_7B => "SmolLM2-1.7B-Instruct-Q4_0.gguf",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            LlmModelId::Gemma3_270M => "Fastest, ~150 MB (removed — crashes on bundled llama.cpp)",
            LlmModelId::SmolLM2_360M => "Fast, ~200 MB GGUF (Q4)",
            LlmModelId::Qwen2_5_0_5B => "Balanced, ~300 MB GGUF (Q4)",
            LlmModelId::TinyLlama_1_1B => "Good quality, ~650 MB GGUF (Q4)",
            LlmModelId::SmolLM2_1_7B => "Best quality, ~950 MB GGUF (Q4)",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmSettings {
    pub enabled: bool,
    pub model: LlmModelId,
    pub max_tokens: u32,
    pub temperature: f32,
    #[serde(default)]
    pub prompt_style: PromptStyle,
    #[serde(default)]
    pub custom_prompt: String,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: u32,
}

const fn default_chunk_size() -> u32 { 500 }

impl Default for LlmSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            model: LlmModelId::SmolLM2_360M,
            max_tokens: 512,
            temperature: 0.0,
            prompt_style: PromptStyle::Smart,
            custom_prompt: String::new(),
            chunk_size: 500,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionSettings {
    pub enabled: bool,
    pub entropy_threshold: f32,
    pub logprob_threshold: f32,
    pub max_reruns: u32,
}

impl Default for CorrectionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            entropy_threshold: 2.0,
            logprob_threshold: -1.5,
            max_reruns: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub audio: AudioSettings,
    pub format: FormatSettings,
    pub processing: ProcessingSettings,
    pub language: String,
    pub model_path: String,
    pub model_size: ModelSize,
    pub input_device_name: Option<String>,
    pub auto_save: bool,
    pub llm: LlmSettings,
    pub correction: CorrectionSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            audio: AudioSettings::default(),
            format: FormatSettings::default(),
            processing: ProcessingSettings::default(),
            language: "auto".to_string(),
            model_path: String::new(),
            model_size: ModelSize::Base,
            input_device_name: None,
            auto_save: true,
            llm: LlmSettings::default(),
            correction: CorrectionSettings::default(),
        }
    }
}

impl Settings {
    pub fn load() -> Self {
        let path = match settings_path() {
            Some(p) => p,
            None => return Self::default(),
        };

        match std::fs::read_to_string(&path) {
            Ok(data) => match serde_json::from_str(&data) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to parse settings: {e}, using defaults");
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) {
        let path = match settings_path() {
            Some(p) => p,
            None => return,
        };

        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        match serde_json::to_string_pretty(self) {
            Ok(data) => {
                if let Err(e) = std::fs::write(&path, &data) {
                    eprintln!("Failed to save settings: {e}");
                }
            }
            Err(e) => eprintln!("Failed to serialize settings: {e}"),
        }
    }

    pub fn resolved_model_path(&self) -> PathBuf {
        if !self.model_path.is_empty() {
            return PathBuf::from(&self.model_path);
        }
        let model_file = format!("ggml-{}.bin", self.model_size.name());
        crate::paths::whisper_model_path(&model_file)
    }
}

fn settings_path() -> Option<PathBuf> {
    let base = dirs::config_dir().or_else(|| dirs::data_dir())?;
    Some(base.join("fluency").join("settings.json"))
}
