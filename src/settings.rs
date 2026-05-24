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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatSettings {
    pub auto_capitalize: bool,
    pub auto_punctuate: bool,
    pub remove_fillers: bool,
    pub fix_corrections: bool,
    pub lm_correction: bool,
    pub fillers: Vec<String>,
    pub correction_markers: Vec<String>,
}

impl Default for FormatSettings {
    fn default() -> Self {
        Self {
            auto_capitalize: true,
            auto_punctuate: true,
            remove_fillers: false,
            fix_corrections: false,
            lm_correction: true,
            fillers: vec![
                "um".into(), "uh".into(), "er".into(), "ah".into(),
                "like".into(), "you know".into(), "i mean".into(),
                "sort of".into(), "kind of".into(), "you know what i mean".into(),
            ],
            correction_markers: vec![
                "or no".into(), "or uh".into(), "or like".into(),
                "i mean".into(), "i meant".into(), "no wait".into(),
                "or rather".into(),
            ],
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
        let data_dir = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
        let model_file = format!("ggml-{}.bin", self.model_size.name());
        data_dir.join("fluency").join(model_file)
    }
}

fn settings_path() -> Option<PathBuf> {
    let base = dirs::config_dir().or_else(|| dirs::data_dir())?;
    Some(base.join("fluency").join("settings.json"))
}
