use std::collections::HashMap;
use std::sync::OnceLock;

static LM: OnceLock<BigramLm> = OnceLock::new();

fn lm() -> &'static BigramLm {
    LM.get_or_init(|| {
        let corpus = include_str!("corpus.txt");
        BigramLm::from_corpus(corpus)
    })
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    let mut prev = (0..=m).collect::<Vec<_>>();
    let mut curr = vec![0; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

fn normalize(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-' && c != '_')
        .to_lowercase()
}

fn detect_language(text: &str) -> &'static str {
    let de_indicators = [
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einer", "einem",
        "ist", "sind", "wird", "werden", "hat", "hast", "haben", "nicht", "und", "oder",
        "aber", "mit", "von", "für", "auf", "bei", "nach", "aus", "durch", "über", "um",
        "zur", "zum", "bin", "bist", "wir", "ihr", "sie", "ich", "du", "er", "sie", "es",
        "man", "kann", "können", "muss", "müssen", "soll", "sollen", "will", "wollen",
        "sein", "ihre", "ihr", "seine", "sein", "mein", "dein", "kein", "keine",
        "diese", "dieser", "dieses", "jenes", "jener", "solche",
        "wenn", "weil", "dass", "da", "ob", "als", "wie", "dann",
        "sehr", "auch", "nur", "schon", "noch", "immer", "wieder",
        "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn",
        "danke", "bitte", "hallo", "tschüss", "guten", "morgen", "abend", "nacht",
        "tags", "woche", "monat", "jahr", "heute", "morgen", "gestern",
        "oben", "unten", "links", "rechts", "vorne", "hinten", "innen", "außen",
        "groß", "klein", "schnell", "langsam", "gut", "schlecht", "neu", "alt",
        "wichtig", "möglich", "verschieden", "einfach", "schwierig", "richtig", "falsch",
        "arbeit", "schule", "haus", "buch", "auto", "stadt", "land", "welt",
        "sicherheit", "leistung", "entwicklung", "forschung", "unternehmen",
        "system", "programm", "daten", "information", "funktion", "methode",
        "version", "software", "hardware", "netzwerk", "speicher", "prozessor",
    ];

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return "en";
    }

    let de_count = words
        .iter()
        .filter(|w| de_indicators.contains(&normalize(w).as_str()))
        .count();

    if de_count as f64 > words.len() as f64 * 0.05 {
        "de"
    } else {
        "en"
    }
}

pub struct BigramLm {
    word_to_id: HashMap<String, u32>,
    #[allow(dead_code)]
    id_to_word: Vec<String>,
    bigram_probs: Vec<Vec<(u32, f32)>>,
    unigram_probs: Vec<f32>,
}

impl BigramLm {
    fn from_corpus(corpus: &str) -> Self {
        let mut raw_counts: HashMap<(u32, u32), u32> = HashMap::new();
        let mut unigram_counts: HashMap<u32, u32> = HashMap::new();
        let mut word_to_id: HashMap<String, u32> = HashMap::new();
        let mut id_to_word: Vec<String> = Vec::new();

        let mut next_id = |w: &str| -> u32 {
            let id = word_to_id.len() as u32;
            *word_to_id.entry(w.to_string()).or_insert_with(|| {
                id_to_word.push(w.to_string());
                id
            })
        };

        let sos = next_id("<s>");
        let eos = next_id("</s>");
        let _unk = next_id("<unk>");

        let total_start = std::time::Instant::now();

        for line in corpus.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("====") {
                continue;
            }

            let mut prev = sos;
            *unigram_counts.entry(sos).or_insert(0) += 1;

            for raw in line.split_whitespace() {
                let w = normalize(raw);
                if w.is_empty() {
                    continue;
                }
                let id = next_id(&w);
                *unigram_counts.entry(id).or_insert(0) += 1;
                *raw_counts.entry((prev, id)).or_insert(0) += 1;
                prev = id;
            }

            *raw_counts.entry((prev, eos)).or_insert(0) += 1;
            *unigram_counts.entry(eos).or_insert(0) += 1;
        }

        let vocab_size = word_to_id.len() as f32;
        let smoothing = 0.1f32;

        let unigram_total: u32 = unigram_counts.values().sum();
        let unigram_probs: Vec<f32> = (0..vocab_size as u32)
            .map(|id| {
                let count = unigram_counts.get(&id).copied().unwrap_or(0) as f32;
                (count + smoothing) / (unigram_total as f32 + smoothing * vocab_size)
            })
            .collect();

        let mut bigram_probs: Vec<Vec<(u32, f32)>> = Vec::with_capacity(vocab_size as usize);
        for i in 0..vocab_size as u32 {
            let ctx_count = unigram_counts.get(&i).copied().unwrap_or(0) as f32;
            let mut row: Vec<(u32, f32)> = (0..vocab_size as u32)
                .filter_map(|j| {
                    let count = raw_counts.get(&(i, j)).copied().unwrap_or(0) as f32;
                    if count > 0.0 {
                        let prob = (count + smoothing) / (ctx_count + smoothing * vocab_size);
                        Some((j, prob))
                    } else {
                        None
                    }
                })
                .collect();
            row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            bigram_probs.push(row);
        }

        eprintln!(
            "LM built: {} words, {} bigrams in {:?}",
            vocab_size as u32,
            raw_counts.len(),
            total_start.elapsed()
        );

        Self {
            word_to_id,
            id_to_word,
            bigram_probs,
            unigram_probs,
        }
    }

    fn id(&self, w: &str) -> Option<u32> {
        self.word_to_id.get(w).copied()
    }

    fn unigram_prob(&self, id: u32) -> f32 {
        self.unigram_probs
            .get(id as usize)
            .copied()
            .unwrap_or(0.0)
    }

    fn bigram_log_prob(&self, prev: u32, word: u32) -> f32 {
        let row = self
            .bigram_probs
            .get(prev as usize)
            .map(|r| {
                r.iter()
                    .find(|(id, _)| *id == word)
                    .map(|(_, p)| *p)
                    .unwrap_or_else(|| self.unigram_prob(word))
            })
            .unwrap_or_else(|| self.unigram_prob(word));

        if row <= 0.0 {
            -20.0
        } else {
            row.ln()
        }
    }

    fn sentence_log_prob(&self, words: &[u32]) -> f32 {
        if words.is_empty() {
            return f32::NEG_INFINITY;
        }
        let sos = self.id("<s>").unwrap_or(0);
        let eos = self.id("</s>").unwrap_or(0);
        let mut lp = self.bigram_log_prob(sos, words[0]);
        for i in 1..words.len() {
            lp += self.bigram_log_prob(words[i - 1], words[i]);
        }
        lp += self.bigram_log_prob(words[words.len() - 1], eos);
        lp
    }

    pub fn known(&self, word: &str) -> bool {
        let n = normalize(word);
        self.word_to_id.contains_key(&n)
    }

    pub fn suggest(&self, text: &str) -> String {
        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| {
                // Preserve original capitalization for output
                w.to_string()
            })
            .collect();

        if words.is_empty() {
            return text.to_string();
        }

        let normalized: Vec<String> = words.iter().map(|w| normalize(w)).collect();

        let ids: Vec<Option<u32>> = normalized.iter().map(|w| self.id(w)).collect();

        let sos = self.id("<s>").unwrap_or(0);
        let _eos = self.id("</s>").unwrap_or(0);

        let mut best_words = words.clone();
        let mut best_ids: Vec<u32> = ids.iter().map(|o| o.unwrap_or(sos)).collect();
        let mut best_score = self.sentence_log_prob(&best_ids);

        let mut improved = false;

        for i in 0..words.len() {
            if normalized[i].is_empty() {
                continue;
            }

            let orig_id = ids[i];
            let orig_norm = &normalized[i];
            let is_known_common = orig_id
                .map(|id| self.unigram_prob(id) > 5e-5)
                .unwrap_or(false);

            let mut candidates: Vec<(String, u32, f32)> = Vec::new();

            // Always add original word as baseline
            if let Some(id) = orig_id {
                candidates.push((words[i].clone(), id, 0.0));
            }

            // Common corrections (always checked, even for high-freq words)
            for (wrong, right) in COMMON_CORRECTIONS {
                if orig_norm == wrong {
                    if let Some(id) = self.id(right) {
                        candidates.push((right.to_string(), id, 0.5));
                    }
                }
            }
            for (wrong, right) in COMMON_DE_CORRECTIONS {
                if orig_norm == wrong {
                    if let Some(id) = self.id(right) {
                        candidates.push((right.to_string(), id, 0.5));
                    }
                }
            }

            // Edit-distance candidates only for unknown or rare words
            if !is_known_common {
                let max_edits = if orig_norm.len() <= 4 { 1 } else { 2 };
                for (vocab_word, &vocab_id) in &self.word_to_id {
                    let d = edit_distance(orig_norm, vocab_word);
                    if d == 0 || d > max_edits {
                        continue;
                    }
                    candidates.push((vocab_word.clone(), vocab_id, d as f32));
                }
            }

            if candidates.is_empty() {
                continue;
            }

            candidates.sort_by(|a, b| {
                let freq_a = self.unigram_prob(a.1);
                let freq_b = self.unigram_prob(b.1);
                freq_b
                    .partial_cmp(&freq_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(5);

            for (cand_word, cand_id, _) in &candidates {
                let mut test_ids = best_ids.clone();
                test_ids[i] = *cand_id;
                let score = self.sentence_log_prob(&test_ids);
                if score > best_score {
                    best_score = score;
                    best_ids = test_ids;
                    best_words[i] = cand_word.clone();
                    improved = true;
                }
            }
        }

        if !improved {
            // Even if no word-level improvements, ensure common fixes
            let mut result = words.join(" ");
            result = fix_punctuation_spacing(&result);
            return result;
        }

        // Restore original capitalization where possible
        let mut result = String::new();
        for (i, w) in best_words.iter().enumerate() {
            if words[i].starts_with(|c: char| c.is_uppercase()) {
                if let Some(c) = w.chars().next() {
                    let uppercased: String = c.to_uppercase().collect();
                    result.push_str(&uppercased);
                    result.push_str(&w[c.len_utf8()..]);
                } else {
                    result.push_str(w);
                }
            } else {
                result.push_str(w);
            }
            if i < best_words.len() - 1 {
                result.push(' ');
            }
        }

        result = fix_punctuation_spacing(&result);
        result
    }
}

fn fix_punctuation_spacing(text: &str) -> String {
    let mut result = text.to_string();
    result = result.replace(" .", ".");
    result = result.replace(" ,", ",");
    result = result.replace(" !", "!");
    result = result.replace(" ?", "?");
    result = result.replace(" ;", ";");
    result = result.replace(" :", ":");
    result = result.replace(" 's", "'s");
    result = result.replace(" 't", "'t");
    result = result.replace(" 'm", "'m");
    result = result.replace(" 're", "'re");
    result = result.replace(" 've", "'ve");
    result = result.replace(" 'll", "'ll");
    result = result.replace(" 'd", "'d");
    result
}

const COMMON_CORRECTIONS: &[(&str, &str)] = &[
    ("there", "their"),
    ("their", "there"),
    ("theyre", "they're"),
    ("theres", "there's"),
    ("its", "it's"),
    ("dont", "don't"),
    ("cant", "can't"),
    ("wont", "won't"),
    ("didnt", "didn't"),
    ("isnt", "isn't"),
    ("arent", "aren't"),
    ("wasnt", "wasn't"),
    ("werent", "weren't"),
    ("havent", "haven't"),
    ("hasnt", "hasn't"),
    ("hadnt", "hadn't"),
    ("couldnt", "couldn't"),
    ("wouldnt", "wouldn't"),
    ("shouldnt", "shouldn't"),
    ("doesnt", "doesn't"),
    ("im", "i'm"),
    ("ive", "i've"),
    ("id", "i'd"),
    ("ill", "i'll"),
    ("youre", "you're"),
    ("youve", "you've"),
    ("youd", "you'd"),
    ("youll", "you'll"),
    ("hes", "he's"),
    ("shes", "she's"),
    ("were", "we're"),
    ("theyre", "they're"),
    ("theyll", "they'll"),
    ("theyd", "they'd"),
    ("wanna", "want to"),
    ("gonna", "going to"),
    ("gotta", "got to"),
    ("lemme", "let me"),
    ("gimme", "give me"),
    ("kinda", "kind of"),
    ("sorta", "sort of"),
    ("lotsa", "lots of"),
    ("outta", "out of"),
    ("cmon", "come on"),
    ("dunno", "don't know"),
    ("gimme", "give me"),
    ("shouldve", "should have"),
    ("couldve", "could have"),
    ("wouldve", "would have"),
    ("mustve", "must have"),
    ("mightve", "might have"),
    ("thats", "that's"),
    ("whats", "what's"),
    ("whos", "who's"),
    ("wheres", "where's"),
    ("whens", "when's"),
    ("whys", "why's"),
    ("hows", "how's"),
    ("lets", "let's"),
];

const COMMON_DE_CORRECTIONS: &[(&str, &str)] = &[
    ("hast", "hat"),
    ("bist", "ist"),
    ("seid", "sind"),
    ("habt", "haben"),
    ("koennen", "können"),
    ("muessen", "müssen"),
    ("woerter", "wörter"),
    ("u", "und"),
    ("n", "und"),
    ("is", "ist"),
    ("sind", "ist"),
    ("nich", "nicht"),
    ("nit", "nicht"),
    ("was", "das"),
    ("wat", "was"),
    ("sin", "sind"),
    ("bin", "ist"),
    ("nen", "einen"),
    ("kein", "nicht"),
    ("muss", "muss"),
    ("kann", "kann"),
    ("wird", "wird"),
    ("wuerde", "würde"),
    ("haette", "hätte"),
    ("waere", "wäre"),
    ("koennte", "könnte"),
    ("sollte", "sollte"),
    ("musste", "musste"),
    ("durfte", "durfte"),
    ("moechte", "möchte"),
    ("danke", "danke"),
    ("bitte", "bitte"),
    ("tschuss", "tschüss"),
    ("gruess", "grüß"),
    ("schon", "schon"),
    ("noch", "noch"),
    ("mal", "einmal"),
    ("grad", "gerade"),
    ("gerade", "gerade"),
    ("eigentlich", "eigentlich"),
    ("natürlich", "natürlich"),
    ("vielleicht", "vielleicht"),
    ("nämlich", "nämlich"),
    ("einfach", "einfach"),
    ("halt", "halt"),
    ("doch", "doch"),
    ("ja", "ja"),
    ("nein", "nein"),
    ("nix", "nichts"),
];

pub fn correct_text(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return String::new();
    }
    lm().suggest(text)
}

pub fn score_text(text: &str) -> f32 {
    let lm = lm();
    let words: Vec<String> = text.split_whitespace().map(|w| normalize(w)).collect();
    let ids: Vec<u32> = words
        .iter()
        .filter_map(|w| lm.id(w))
        .collect();
    if ids.is_empty() {
        return f32::NEG_INFINITY;
    }
    lm.sentence_log_prob(&ids)
}

pub fn is_word_known(word: &str) -> bool {
    lm().known(word)
}

pub fn detect_text_language(text: &str) -> &'static str {
    detect_language(text)
}
