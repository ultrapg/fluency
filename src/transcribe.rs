use whisper_rs::{FullParams, SamplingStrategy, WhisperVadParams};

use crate::settings::ProcessingSettings;

type ProgressCb = Box<dyn FnMut(i32)>;

pub fn transcribe(
    model_path: &str,
    audio: &[f32],
    language: Option<&str>,
    processing: &ProcessingSettings,
) -> anyhow::Result<String> {
    transcribe_impl(model_path, audio, language, processing, None::<ProgressCb>)
}

pub fn transcribe_with_progress<F: FnMut(i32) + 'static>(
    model_path: &str,
    audio: &[f32],
    language: Option<&str>,
    processing: &ProcessingSettings,
    progress: F,
) -> anyhow::Result<String> {
    transcribe_impl(model_path, audio, language, processing, Some(Box::new(progress)))
}

fn transcribe_impl(
    model_path: &str,
    audio: &[f32],
    language: Option<&str>,
    processing: &ProcessingSettings,
    progress: Option<ProgressCb>,
) -> anyhow::Result<String> {
    let ctx = whisper_rs::WhisperContext::new_with_params(
        model_path,
        whisper_rs::WhisperContextParameters::default(),
    )?;

    let strategy = if processing.sampling_strategy == "greedy" {
        SamplingStrategy::Greedy {
            best_of: processing.best_of.max(1),
        }
    } else {
        SamplingStrategy::BeamSearch {
            beam_size: processing.beam_size.max(1),
            patience: processing.patience,
        }
    };
    let mut params = FullParams::new(strategy);

    if processing.n_threads > 0 {
        params.set_n_threads(processing.n_threads);
    }
    if processing.n_max_text_ctx > 0 {
        params.set_n_max_text_ctx(processing.n_max_text_ctx);
    }
    params.set_no_timestamps(true);
    params.set_temperature(processing.temperature);
    params.set_temperature_inc(processing.temperature_inc);
    params.set_suppress_blank(processing.suppress_blank);
    params.set_suppress_nst(processing.suppress_nst);
    params.set_no_speech_thold(processing.no_speech_thold);
    params.set_length_penalty(processing.length_penalty);
    params.set_entropy_thold(processing.entropy_thold);
    params.set_logprob_thold(processing.logprob_thold);
    params.set_no_context(processing.no_context);
    params.set_single_segment(processing.single_segment);
    params.set_audio_ctx(processing.audio_ctx);
    params.set_debug_mode(processing.debug_mode);
    params.set_max_tokens(processing.max_tokens);
    params.set_language(language);
    params.set_initial_prompt(&processing.initial_prompt);

    if processing.vad_enabled {
        params.set_vad_model_path(Some(model_path));
        let mut vp = WhisperVadParams::new();
        vp.set_threshold(processing.vad_threshold);
        vp.set_min_speech_duration(processing.vad_min_speech_duration_ms);
        vp.set_min_silence_duration(processing.vad_min_silence_duration_ms);
        vp.set_max_speech_duration(processing.vad_max_speech_duration_s);
        vp.set_speech_pad(processing.vad_speech_pad_ms);
        vp.set_samples_overlap(processing.vad_samples_overlap);
        params.set_vad_params(vp);
        params.enable_vad(true);
    }

    params.set_progress_callback_safe::<Option<ProgressCb>, ProgressCb>(progress);

    let mut state = ctx.create_state()?;
    state.full(params, audio)?;

    let text: String = state
        .as_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(" ");

    Ok(text)
}
