use std::collections::HashSet;
use std::num::NonZeroU32;
use crate::lm;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token_type::LlamaTokenAttr;

const HEDGE_WORDS: [&str; 7] = [
    "maybe", "about", "roughly", "approximately", "around", "almost", "nearly",
];

pub struct Formatter {
    pub capitalize: bool,
    pub punctuate: bool,
    pub remove_fillers: bool,
    pub fix_corrections: bool,
    pub lm_correct: bool,
    fillers: Vec<String>,
    correction_markers: Vec<String>,
    question_starts: HashSet<&'static str>,
    pub llm: crate::settings::LlmSettings,
}

impl Default for Formatter {
    fn default() -> Self {
        Self::new()
    }
}

impl Formatter {
    pub fn new() -> Self {
        let defaults = crate::settings::FormatSettings::default();
        Self {
            capitalize: true,
            punctuate: true,
            remove_fillers: false,
            fix_corrections: false,
            lm_correct: true,
            fillers: defaults.fillers.clone(),
            correction_markers: defaults.correction_markers.clone(),
            question_starts: HashSet::from([
                "what", "when", "where", "why", "how", "who", "whom", "whose", "which",
                "do", "does", "did", "is", "are", "was", "were",
                "can", "could", "will", "would", "shall", "should",
                "have", "has", "had", "may", "might", "am",
            ]),
            llm: crate::settings::LlmSettings::default(),
        }
    }

    pub fn with_fillers(mut self, remove: bool) -> Self {
        self.remove_fillers = remove;
        self
    }

    pub fn with_lm_correction(mut self, enable: bool) -> Self {
        self.lm_correct = enable;
        self
    }

    pub fn with_llm(mut self, settings: &crate::settings::LlmSettings) -> Self {
        self.llm = settings.clone();
        self
    }

    pub fn with_settings(mut self, settings: &crate::settings::FormatSettings) -> Self {
        self.capitalize = settings.auto_capitalize;
        self.punctuate = settings.auto_punctuate;
        self.remove_fillers = settings.remove_fillers;
        self.fix_corrections = settings.fix_corrections;
        self.lm_correct = settings.lm_correction;
        self.fillers = settings.fillers.clone();
        self.correction_markers = settings.correction_markers.clone();
        self
    }

    pub fn format(&self, text: &str) -> String {
        if self.llm.enabled {
            self.format_with_llm(text)
        } else {
            self.format_classic(text)
        }
    }

    pub fn format_classic(&self, text: &str) -> String {
        let text = text.trim();
        if text.is_empty() {
            return String::new();
        }

        let text = if self.remove_fillers || self.fix_corrections {
            let text = if self.remove_fillers {
                self.remove_fillers(text)
            } else {
                text.to_string()
            };
            if self.fix_corrections {
                self.remove_corrections(&text)
            } else {
                text
            }
        } else {
            text.to_string()
        };
        let text = text.trim();

        let sentences = self.split_sentences(&text);
        let formatted: Vec<String> = sentences
            .into_iter()
            .map(|s| self.format_sentence(s.trim()))
            .filter(|s| !s.is_empty())
            .collect();

        let mut result = formatted.join(" ");

        result = result
            .replace(" i ", " I ")
            .replace(" i'm", " I'm")
            .replace(" i'll", " I'll")
            .replace(" i've", " I've")
            .replace(" i'd", " I'd");

        if self.lm_correct {
            result = lm::correct_text(&result);
        }

        result
    }

    fn split_sentences<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut sentences = Vec::new();
        let mut start = 0;

        for (i, c) in text.char_indices() {
            if matches!(c, '.' | '!' | '?') {
                if i + 1 < text.len() {
                    let next = text[i + 1..].trim_start();
                    if next.starts_with(|c: char| c.is_uppercase()) {
                        sentences.push(&text[start..=i]);
                        start = i + 1;
                    }
                } else {
                    sentences.push(&text[start..=i]);
                    start = text.len();
                }
            }
        }

        if start < text.len() {
            let remaining = &text[start..];
            if !remaining.trim().is_empty() {
                sentences.push(remaining);
            }
        }

        if sentences.is_empty() {
            sentences.push(text);
        }

        sentences
    }

    fn format_sentence(&self, sentence: &str) -> String {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            return String::new();
        }

        let chars: Vec<char> = sentence.chars().collect();
        let mut result = String::new();

        for (i, c) in chars.iter().enumerate() {
            if i == 0 && self.capitalize {
                result.push(c.to_uppercase().next().unwrap_or(*c));
            } else {
                result.push(*c);
            }
        }

        if !self.punctuate {
            return result.trim_end().to_string();
        }

        let mut out = result.trim_end().to_string();

        if out.is_empty() {
            return out;
        }

        let last = out.chars().last().unwrap();
        match last {
            '.' | '!' | '?' | '\n' => {}
            ',' | ';' | ':' => {
                out.pop();
                if self.is_question(&out) {
                    out.push('?');
                } else {
                    out.push('.');
                }
            }
            _ => {
                if self.is_question(&out) {
                    out.push('?');
                } else {
                    out.push('.');
                }
            }
        }

        out
    }

    fn is_question(&self, text: &str) -> bool {
        let lower = text.trim().to_lowercase();
        for start in &self.question_starts {
            if lower.starts_with(start) && {
                let after = &lower[start.len()..];
                after.is_empty() || after.starts_with(' ') || after.starts_with('\'')
            } {
                return true;
            }
        }
        false
    }

    fn remove_fillers(&self, text: &str) -> String {
        let mut result = text.to_string();
        for filler in &self.fillers {
            let patterns = [
                format!(" {filler} "),
                format!(" {filler},"),
                format!(" {filler}."),
                format!(" {filler}?"),
                format!("{filler} "),
            ];
            for pattern in &patterns {
                result = result.replace(pattern.as_str(), " ");
            }
        }

        let mut cleaned = String::with_capacity(result.len());
        let mut prev_space = false;
        for c in result.chars() {
            if c == ' ' {
                if !prev_space {
                    cleaned.push(c);
                }
                prev_space = true;
            } else {
                cleaned.push(c);
                prev_space = false;
            }
        }

        cleaned
    }

    fn remove_corrections(&self, text: &str) -> String {
        let text = text.trim();
        if text.is_empty() {
            return text.to_string();
        }

        let mut result = text.to_string();

        loop {
            let mut best_pos: Option<usize> = None;
            let mut best_marker_len = 0;

            for marker in &self.correction_markers {
                if let Some(pos) = result[..result.len().saturating_sub(1)].rfind(marker) {
                    let after = &result[pos + marker.len()..];
                    if !after.trim_start().is_empty() {
                        if best_pos.map_or(true, |best| pos > best) {
                            best_pos = Some(pos);
                            best_marker_len = marker.len();
                        }
                    }
                }
            }

            let pos = match best_pos {
                Some(p) => p,
                None => break,
            };

            let before = &result[..pos];
            let before_trimmed = before.trim_end();

            let wrong_start = before_trimmed.rfind(' ').map(|p| p + 1).unwrap_or(0);

            let hedge_start = if wrong_start > 0 {
                let before_wrong = &before_trimmed[..wrong_start.saturating_sub(1)];
                let last_word_start = before_wrong.rfind(' ').map(|p| p + 1).unwrap_or(0);
                let last_word = &before_wrong[last_word_start..];
                if HEDGE_WORDS.contains(&last_word) {
                    Some(last_word_start)
                } else {
                    None
                }
            } else {
                None
            };

            let remove_start = hedge_start.unwrap_or(wrong_start);

            let prefix = &result[..remove_start].trim_end();

            let after = &result[pos + best_marker_len..];
            let after_trimmed = after.trim_start();

            let corrected_end = after_trimmed.find(' ').unwrap_or(after_trimmed.len());
            let corrected = &after_trimmed[..corrected_end];
            let suffix = after_trimmed[corrected_end..].trim_start();

            if prefix.is_empty() {
                if suffix.is_empty() {
                    result = corrected.to_string();
                } else {
                    result = format!("{} {}", corrected, suffix);
                }
            } else if suffix.is_empty() {
                result = format!("{} {}", prefix, corrected);
            } else {
                result = format!("{} {} {}", prefix, corrected, suffix);
            }
        }

        result
    }

    pub fn format_with_llm(&self, text: &str) -> String {
        let text = text.trim();
        if text.is_empty() {
            return String::new();
        }

        let model_path = crate::paths::llm_model_path(self.llm.model.gguf_filename());
        if !model_path.exists() {
            if let Err(e) = crate::model::ensure_llm_downloaded(self.llm.model) {
                eprintln!("Error downloading LLM model: {e}");
                return self.format_classic(text);
            }
        }

        let mut backend = match LlamaBackend::init() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Failed to initialize llama.cpp backend: {e}");
                return self.format_classic(text);
            }
        };
        backend.void_logs();

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(999)
            .with_use_mmap(true);

        let model = match LlamaModel::load_from_file(&backend, &model_path, &model_params) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load LLM model: {e}");
                return self.format_classic(text);
            }
        };

        let sentences = self.split_sentences(text);
        if sentences.is_empty() {
            return String::new();
        }

        let mut chunks = Vec::new();
        let mut current_chunk_sentences: Vec<&str> = Vec::new();
        let mut current_chunk_len = 0;
        let mut prev_sentence: Option<String> = None;

        for &sentence in &sentences {
            let sentence_len = sentence.len();
            if current_chunk_len + sentence_len > self.llm.chunk_size as usize && !current_chunk_sentences.is_empty() {
                chunks.push((current_chunk_sentences.clone(), prev_sentence.clone()));
                prev_sentence = Some(current_chunk_sentences.last().unwrap().to_string());
                current_chunk_sentences.clear();
                current_chunk_len = 0;
            }
            current_chunk_sentences.push(sentence);
            current_chunk_len += sentence_len;
        }
        if !current_chunk_sentences.is_empty() {
            chunks.push((current_chunk_sentences, prev_sentence));
        }

        let mut formatted_text = String::new();

        for (chunk_sentences, prev_sent) in chunks {
            let original_chunk_text = chunk_sentences.join(" ");
            let system = system_prompt(self.llm.prompt_style, &self.fillers);
            let user = format_chunk_prompt(&chunk_sentences, prev_sent.as_deref());
            let formatted_prompt = format_prompt(self.llm.model.template(), &system, &user);

            let result = run_inference_helper(&model, &backend, &formatted_prompt, self.llm.temperature, self.llm.max_tokens);
            match result {
                Ok(generated) => {
                    let cleaned = clean_llm_response(&generated);
                    if validate_formatted_chunk(&original_chunk_text, &cleaned, &self.fillers) {
                        formatted_text.push_str(&cleaned);
                        formatted_text.push(' ');
                    } else {
                        formatted_text.push_str(&original_chunk_text);
                        formatted_text.push(' ');
                    }
                }
                Err(e) => {
                    eprintln!("LLM inference error: {e}");
                    formatted_text.push_str(&original_chunk_text);
                    formatted_text.push(' ');
                }
            }
        }

        let mut final_result = formatted_text.trim().to_string();

        if self.lm_correct {
            final_result = lm::correct_text(&final_result);
        }

        final_result
    }
}

fn system_prompt(style: crate::settings::PromptStyle, fillers: &[String]) -> String {
    let filler_list = fillers.join(", ");
    match style {
        crate::settings::PromptStyle::Clean => format!(
            "You are a transcription assistant. Your job is to format the input speech transcription into clean paragraphs. \
             Add proper capitalization, periods, commas, and paragraph breaks. \
             Remove filler words: {}. \
             CRITICAL: Do not rephrase, rewrite, or add any new words. Keep all other words exactly as they are.",
            filler_list
        ),
        crate::settings::PromptStyle::BulletPoints => format!(
            "You are a transcription assistant. Your job is to format the input speech transcription as a bullet-point list. \
             Each sentence or key point must start with a new line and a '- ' prefix. \
             Add proper capitalization and punctuation. Remove filler words: {}. \
             CRITICAL: Do not rephrase, rewrite, or add any new words. Keep all other words exactly as they are.",
            filler_list
        ),
        crate::settings::PromptStyle::Smart => format!(
            "You are a transcription assistant. Your job is to improve capitalization, punctuation, and paragraph breaks of the transcription. \
             Remove filler words: {}. \
             CRITICAL: Do not rephrase, rewrite, or change any words. Keep all sentences exactly as spoken.",
            filler_list
        ),
        crate::settings::PromptStyle::Minimal => {
            "You are a transcription assistant. Capitalize the first letter and add an ending period. \
             Do not add, remove, or change any other words."
                .to_string()
        }
    }
}

fn format_chunk_prompt(sentences: &[&str], prev_sentence: Option<&str>) -> String {
    let mut prompt = String::new();
    if let Some(prev) = prev_sentence {
        prompt.push_str(&format!("Context: {}\n", prev));
    }
    prompt.push_str("Text to format: ");
    prompt.push_str(&sentences.join(" "));
    prompt
}

fn format_prompt(template: &str, system: &str, user: &str) -> String {
    match template {
        "qwen" => format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system, user
        ),
        "llama" => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            system, user
        ),
        _ => format!("System: {}\nUser: {}\nAssistant: ", system, user),
    }
}

fn clean_llm_response(text: &str) -> String {
    let mut cleaned = text.to_string();
    while let Some(start) = cleaned.find("<think>") {
        if let Some(end) = cleaned.find("</think>") {
            if end > start {
                cleaned.drain(start..end + 8);
            } else {
                cleaned.drain(start..);
            }
        } else {
            cleaned.drain(start..);
            break;
        }
    }
    while let Some(start) = cleaned.find("<thinking>") {
        if let Some(end) = cleaned.find("</thinking>") {
            if end > start {
                cleaned.drain(start..end + 11);
            } else {
                cleaned.drain(start..);
            }
        } else {
            cleaned.drain(start..);
            break;
        }
    }
    cleaned.trim().to_string()
}

fn validate_formatted_chunk(original: &str, formatted: &str, fillers: &[String]) -> bool {
    let orig_words: Vec<String> = original
        .split_whitespace()
        .map(|w| normalize_word(w))
        .filter(|w| !w.is_empty())
        .collect();
    let gen_words: Vec<String> = formatted
        .split_whitespace()
        .map(|w| normalize_word(w))
        .filter(|w| !w.is_empty())
        .collect();

    let mut i = 0;
    let mut j = 0;

    while i < orig_words.len() && j < gen_words.len() {
        if orig_words[i] == gen_words[j] {
            i += 1;
            j += 1;
        } else if fillers.iter().any(|f| f.to_lowercase() == orig_words[i]) {
            i += 1;
        } else {
            return false;
        }
    }

    while i < orig_words.len() {
        if fillers.iter().any(|f| f.to_lowercase() == orig_words[i]) {
            i += 1;
        } else {
            return false;
        }
    }

    j == gen_words.len()
}

fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

fn run_inference_helper(
    model: &LlamaModel,
    backend: &LlamaBackend,
    prompt_text: &str,
    temperature: f32,
    max_tokens: u32,
) -> anyhow::Result<String> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| anyhow::anyhow!("Failed to create context: {e:?}"))?;

    let tokens = model
        .str_to_token(prompt_text, AddBos::Always)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e:?}"))?;

    let mut batch = LlamaBatch::new(2048, 1);
    for (i, token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch.add(*token, i as i32, &[0], is_last)?;
    }

    ctx.decode(&mut batch).map_err(|e| anyhow::anyhow!("Failed to decode prompt: {e:?}"))?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(temperature),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::greedy(),
    ]);

    let mut n_cur = tokens.len() as i32;
    let mut token = sampler.sample(&ctx, batch.n_tokens() - 1);
    sampler.accept(token);

    let mut generated_text = String::new();

    if !model.token_attr(token).contains(LlamaTokenAttr::Control) {
        let first_piece = token_to_string_helper(model, token);
        if !first_piece.contains("<|im_end|>") && !first_piece.contains("<|eot_id|>") {
            generated_text.push_str(&first_piece);
        }
    }

    while token != model.token_eos() && n_cur < 2048 && (generated_text.chars().count() as u32) < max_tokens {
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;

        ctx.decode(&mut batch).map_err(|e| anyhow::anyhow!("Failed to decode token: {e:?}"))?;
        token = sampler.sample(&ctx, 0);
        sampler.accept(token);

        if model.token_attr(token).contains(LlamaTokenAttr::Control) {
            break;
        }

        let piece = token_to_string_helper(model, token);
        if piece.contains("<|im_end|>") || piece.contains("<|eot_id|>") {
            break;
        }
        generated_text.push_str(&piece);
    }

    Ok(generated_text)
}

fn token_to_string_helper(model: &LlamaModel, token: LlamaToken) -> String {
    match model.token_to_piece_bytes(token, 32, false, None) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => {
            if let Ok(bytes) = model.token_to_piece_bytes(token, 256, false, None) {
                String::from_utf8_lossy(&bytes).into_owned()
            } else {
                String::new()
            }
        }
    }
}
