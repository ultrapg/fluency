use std::collections::HashSet;

use crate::lm;

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
}
